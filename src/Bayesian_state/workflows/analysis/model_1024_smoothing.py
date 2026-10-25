"""Exact finite-state benchmark and passive retrospective recovery.

Use --phase benchmark|run --config YAML --output NEW [--smoke]. The run phase
reuses registered cases, true parameters and archived shared-model workers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
HELPER = 'src/Bayesian_state/evaluation/particle_smoothing.py'
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1023_execution_capture.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'

# Workers start with the frozen runtime on sys.path. Load opt-in analysis
# helpers by path; scientific model imports keep the archived package first.
import importlib.util


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture = load('smoothing_capture', CAPTURE)
study = capture.study
read, write, digest = study.read, study.write, study.digest
helper = load('smoothing_numeric_helpers', HELPER)


def sources() -> dict:
    return dict(study.sources(study.driver.io), **capture.sources(),
                **{p: digest(ROOT/p) for p in (ENTRY, HELPER)})


def state_means(values: dict, weights: np.ndarray) -> dict:
    """Aggregate target events under a common endpoint's ancestral weights."""
    result = {key: np.einsum('tp,tp->t', weights, values[key])
              for key in ('present', 'high_belief', 'target_belief', 'target_execution')}
    result['source'] = np.stack([np.sum(weights*(values['source'] == k), axis=1) for k in range(3)], axis=1)
    helper.probabilities(result['source'])
    return result


def genealogy_summary(values: dict, parents: np.ndarray, weights: np.ndarray, tracer, prefix: int, lag: int) -> dict:
    """Keep full and prefix posterior ensembles distinct from changing endpoints."""
    n, particles = weights.shape
    if not 1 <= prefix <= n or lag < 0 or parents.shape != (n, particles):
        raise ValueError('Invalid smoothing endpoints')
    arrays = {'filtered_'+k: v for k, v in state_means(values, weights).items()}
    for name, endpoint in [('prefix', prefix), ('full', n)]:
        paths = tracer(parents[:endpoint])
        ancestry = helper.ancestral_weights(paths, weights[endpoint-1])
        selected = {k: v[:endpoint] for k, v in values.items()}
        arrays.update({name+'_'+k: v for k,v in state_means(selected, ancestry['weights']).items()})
        arrays[name+'_unique_ancestors'] = ancestry['unique_ancestors']
        arrays[name+'_ancestral_ess'] = ancestry['ancestral_ess']
        arrays[name+'_indices'] = paths.astype(np.int32)
    # Each lagged row has its own observation endpoint. These rows must not
    # be treated as one common-condition posterior path distribution.
    lagged = {k: [] for k in state_means(values, weights)}; counts = []; ess = []; endpoints = []
    for t in range(n):
        end = min(n, t+lag+1); ancestors = np.arange(particles)
        for a in range(end-2, t-1, -1):
            ancestors = parents[a, ancestors]
        mass = np.bincount(ancestors, weights=weights[end-1], minlength=particles)
        own = state_means({k: v[t:t+1] for k,v in values.items()}, mass[None, :])
        for k in own:
            lagged[k].append(own[k][0])
        counts.append(np.count_nonzero(mass > 0)); ess.append(1./np.square(mass).sum()); endpoints.append(end)
    arrays.update({'lag_'+k: np.asarray(v) for k,v in lagged.items()})
    arrays.update(lag_unique_ancestors=np.array(counts), lag_ancestral_ess=np.array(ess),
                  lag_endpoints=np.array(endpoints))
    return arrays


def worker(file: Path) -> None:
    job = read(file)
    if job['smoothing_sources'] != sources() or job.get('capture_oral'):
        raise ValueError('Require unchanged behavior-only smoothing sources')
    original_load = capture.study.base.load; audit = {}

    def instrument(name, path):
        module = original_load(name, path)
        if path == PF:
            original_filter = module.run_state_model_particle_filter
            def observe(**kwargs):
                existing = kwargs.pop('state_observer')
                collector = helper.TargetGenealogyCollector(job['target_rules'])
                def both(t, phase, snapshot):
                    existing(t, phase, snapshot); collector(t, phase, snapshot)
                result = original_filter(**kwargs, state_observer=both)
                values = collector.arrays(); weights = values.pop('filtered_weights')
                parents = helper.resampling_parents(result.resampling_log, len(weights), job['particles'])
                summary = genealogy_summary(values, parents, weights, module._trace_ancestral_indices,
                    min(job['prefix_trials'], len(weights)), job['fixed_lag'])
                target = Path(job['genealogy_target']); target.parent.mkdir(parents=True, exist_ok=True)
                with target.with_suffix('.npz').open('xb') as stream:
                    np.savez_compressed(stream, **{'particle_'+k: v for k,v in values.items()},
                        filtered_weights=weights, parents=parents, **summary)
                audit.update(trials=len(weights), particles=job['particles'],
                    terminal_timing='after final choice, before final feedback and resampling',
                    last_trial_resampled=bool(result.resampled[-1]),
                    array_sha256=digest(target.with_suffix('.npz')),
                    final_pre_resample_ess=float(1./np.square(weights[-1]).sum()),
                    resampling_steps=int(np.count_nonzero(result.resampled)))
                return result
            module.run_state_model_particle_filter = observe
        return module
    capture.study.base.load = instrument
    try:
        capture.worker(file)
    finally:
        capture.study.base.load = original_load
    if not audit or job['smoothing_sources'] != sources():
        raise ValueError('Incomplete or changing smoothing capture')
    # Online aggregates must match the original collector as well as saved
    # behavioral predictions. This specifically checks choice conditioning.
    with np.load(job['genealogy_target']+'.npz') as own, np.load(job['state_capture_target']+'.npz') as previous:
        for a,b in [('present','target_present'), ('high_belief','target_high_belief'),
                    ('target_belief','target_belief'), ('target_execution','target_execution')]:
            np.testing.assert_allclose(own['filtered_'+a], previous['filtered_'+b], atol=2e-12, rtol=0)
        np.testing.assert_allclose(own['filtered_source'], previous['conditional_decision_sources'], atol=2e-12, rtol=0)
    write(Path(job['genealogy_target']).with_suffix('.json'), dict(status='complete', **audit,
        sources=sources(), case_id=job['recovery_case'], subject=job['case_id'], seeds=job['seeds'],
        prefix_trials=min(job['prefix_trials'],audit['trials']), fixed_lag=job['fixed_lag'],
        original_state_replay=bool(job.get('original_state')), original_score_replay=bool(job.get('original_score')),
        state_capture_sha256=digest(Path(job['state_capture_target']).with_suffix('.npz')),
        oral_used=False, conditioning='known parameters; observed choices and task feedback',
        state_scope='target events and execution identity; complete cognitive states are not retained'))


def brute_hmm(initial: np.ndarray, transitions: np.ndarray, likelihood: np.ndarray) -> dict:
    """Independent complete-path enumeration, restricted to short diagnostics."""
    n, states = likelihood.shape
    if states**n > 1_000_000:
        raise ValueError('Enumeration budget exceeded')
    paths = np.array(list(product(range(states), repeat=n)), dtype=np.int16)
    mass = initial[paths[:, 0]]*likelihood[0, paths[:, 0]]
    for t in range(1,n):
        mass *= transitions[t-1, paths[:, t-1], paths[:, t]]*likelihood[t, paths[:, t]]
    evidence = mass.sum()
    if evidence <= 0:
        raise ValueError('Impossible enumerated observation')
    mass /= evidence
    marginal = np.stack([np.bincount(paths[:,t],weights=mass,minlength=states) for t in range(n)])
    pairs = np.stack([np.bincount(paths[:,t]*states+paths[:,t+1],weights=mass,
                                minlength=states**2).reshape(states,states) for t in range(n-1)]) if n>1 else np.empty((0,states,states))
    return dict(paths=paths, weights=mass, smoothed=marginal, pairs=pairs, log_evidence=float(np.log(evidence)))


def hmm_particles(initial, transitions, likelihood, particles, seed, backend) -> dict:
    """Diagnostic discrete bootstrap filter using the shared resampler/tracer."""
    rng = np.random.default_rng(seed); n, states = likelihood.shape
    particle = rng.choice(states, size=particles, p=initial); w = np.full(particles,1./particles)
    history = []; filtered = []; parents = np.tile(np.arange(particles),(n,1)); terminal = None
    for t in range(n):
        if t:
            uniform = rng.random(particles)
            particle = np.sum(uniform[:,None] > transitions[t-1,particle].cumsum(axis=1),axis=1)
        history.append(particle.copy()); w *= likelihood[t,particle]
        if w.sum() <= 0:
            raise ValueError('Diagnostic filter lost all likelihood support')
        w /= w.sum(); filtered.append(np.bincount(particle,weights=w,minlength=states))
        if t == n-1:
            terminal = w.copy()
        if 1./np.square(w).sum() < .5*particles:
            ancestors = backend.systematic_resample(w,float(rng.random()))
            parents[t] = ancestors; particle = particle[ancestors]; w.fill(1./particles)
    indices = backend._trace_ancestral_indices(parents)
    selected = helper.trace_values(np.array(history),indices)
    smoothed = np.stack([np.bincount(row,weights=terminal,minlength=states) for row in selected])
    ancestry = helper.ancestral_weights(indices,terminal)
    return dict(filtered=np.array(filtered),smoothed=smoothed,
                unique=ancestry['unique_ancestors'],ess=ancestry['ancestral_ess'])


def benchmark(config: Path, output: Path) -> None:
    import pandas as pd
    import yaml
    from ...inference.backends import particle_filter as backend
    if output.exists():
        raise FileExistsError('Use a new exact benchmark directory')
    p=yaml.safe_load(config.read_text()); b=p['benchmark']; output.mkdir(parents=True)
    initial=np.array(b['initial']); belief=np.array(b['belief']); kernel=np.tile(belief,(len(belief),1))
    kernel *= np.exp(b['inertia']*np.eye(len(belief))); kernel /= kernel.sum(axis=1,keepdims=True)
    emissions=helper.probabilities(np.array(b['emissions'])); rows=[]; curves=[]; checks=[]
    for n in b['lengths']:
        transitions=np.tile(kernel,(n-1,1,1)); rng=np.random.default_rng(20261025+n)
        states=np.empty(n,int); choices=np.empty(n,int)
        for t in range(n):
            states[t]=rng.choice(len(initial),p=initial if t==0 else kernel[states[t-1]])
            choices[t]=rng.choice(emissions.shape[1],p=emissions[states[t]])
        likelihood=emissions[:,choices].T; exact=helper.exact_hmm(initial,transitions,likelihood)
        if n<=8:
            brute=brute_hmm(initial,transitions,likelihood)
            for key in ('smoothed','pairs','log_evidence'):
                np.testing.assert_allclose(exact[key],brute[key],atol=2e-13,rtol=0)
            # Independently enumerate each observed prefix to validate filtering.
            for end in range(1,n+1):
                one=brute_hmm(initial,transitions[:end-1],likelihood[:end])
                np.testing.assert_allclose(exact['filtered'][end-1],one['smoothed'][-1],atol=2e-13,rtol=0)
            checks.append(dict(length=n,enumerated_paths=len(brute['paths']),exact_enumeration=True))
        samples=helper.exact_hmm_paths(exact['filtered'],transitions,b['exact_path_samples'],20261025+n)
        sampled=np.stack([np.bincount(samples[:,t],minlength=len(initial))/len(samples) for t in range(n)])
        checks.append(dict(length=n,backward_samples=len(samples),max_marginal_error=float(np.abs(sampled-exact['smoothed']).max())))
        with (output/f'exact_T{n}.npz').open('xb') as stream:
            np.savez_compressed(stream,initial=initial,transitions=transitions,likelihood=likelihood,
                true_states=states,choices=choices,**exact,samples=samples)
        for particles in b['particles']:
            for seed in b['seeds']:
                pf=hmm_particles(initial,transitions,likelihood,particles,seed,backend)
                for mode in ('filtered','smoothed'):
                    error=np.abs(pf[mode]-exact[mode])
                    rows.append(dict(length=n,particles=particles,seed=seed,mode=mode,
                        mean_absolute_error=float(error.mean()),max_absolute_error=float(error.max()),
                        first_unique_ancestors=int(pf['unique'][0]),first_ancestral_ess=float(pf['ess'][0])))
                for t in range(n):
                    curves.append(dict(length=n,particles=particles,seed=seed,trial=t+1,
                        exact_filtered=exact['filtered'][t,2],exact_smoothed=exact['smoothed'][t,2],
                        filtered=pf['filtered'][t,2],smoothed=pf['smoothed'][t,2],
                        unique_ancestors=pf['unique'][t],ancestral_ess=pf['ess'][t]))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False);pd.DataFrame(curves).to_csv(output/'curves.csv',index=False)
    write(output/'completion.json',dict(status='complete',policy=p,checks=checks,sources=sources(),
        config_sha256=digest(config),files={f.name:digest(f) for f in output.iterdir() if f.is_file()},oral_used=False))
    print('Exact finite-state benchmark complete',flush=True)


def run(config: Path, output: Path, smoke: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import os
    import yaml
    from . import model_1023_execution_validation as validation
    from . import model_1024_state_oral_validation as previous
    if output.exists():
        raise FileExistsError('Use a new smoothing run directory')
    p=yaml.safe_load(config.read_text())
    if (p['cases']!=['C004','C005','C006','C007','C008','C010','C013'] or p['particles']!=512
            or p['filter_seeds']!=3 or p['fixed_lag']!=32 or p['oral_status']!='deferred_pending_user_manual_review'):
        raise ValueError('Require registered behavior-only smoothing scope')
    prepared=ROOT/p['previous_prepared']; inference=ROOT/p['previous_inference']
    prep=read(prepared/'protocol.json'); old=read(inference/'protocol.json')
    if (read(prepared/'completion.json')['protocol_sha256']!=digest(prepared/'protocol.json')
            or read(inference/'completion.json')['status']!='complete'):
        raise ValueError('Require complete registered source data')
    output.mkdir(parents=True); jobs=[]; cases=[]
    for cid in p['cases']:
        c=next(c for c in old['cases'] if c['definition']['case_id']==cid)
        row=next(r for r in prep['numerical'] if r['case_id']==cid)
        if smoke and cid not in ('C004','C008','C010'):
            continue
        template=deepcopy(c['definition']['template'])
        if smoke:
            template=row['template']  # short registered prefix; filter smoke uses 8 trials
        case=dict(subject=c['definition']['subject'],template=template,target_rules=row['target_rules'])
        cases.append(dict(case_id=cid,subject=case['subject'],scope=c['definition']['scope'],
            window_start=row['window_start'],window_stop=row['window_stop'],prefix_trials=row['prefix_trials'],
            truth=c['truth'],target_rules=row['target_rules']))
        for index,seed in enumerate(row['seeds'][:1] if smoke else row['seeds']):
            name=f'{cid}_P{2 if smoke else p["particles"]}_S{index}'
            reference=next(r for r in row['references'] if r['particles']==512 and r['seed']==seed)
            job=previous.inference_job(case,c['truth']['candidate'],seed,2 if smoke else p['particles'],
                output,name,smoke,reference=None if smoke else Path(reference['score']).with_suffix('.npz'))
            job.update(recovery_case=cid,smoothing_sources=sources(),fixed_lag=p['fixed_lag'],
                prefix_trials=row['prefix_trials'],genealogy_target=str((output/'genealogy'/name).resolve()),
                source_sha256=study.sources(study.driver.io))
            if not smoke:
                job.update(original_state=reference['state'],original_state_sha256=reference['state_sha256'])
            jobs.append(job)
    workers=1 if smoke else min(p['parallel_budget'],len(os.sched_getaffinity(0)),len(jobs))
    write(output/'protocol.json',dict(policy=p,config_sha256=digest(config),sources=sources(),jobs=jobs,cases=cases,
        inputs={str(prepared/'protocol.json'):digest(prepared/'protocol.json'),str(inference/'protocol.json'):digest(inference/'protocol.json')},
        smoke=smoke,workers=workers,oral_used=False))
    original_entry=validation.ENTRY;validation.ENTRY=ENTRY;started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            validation.NumericalQueue(pool).execute(jobs,output,workers)
    finally:
        validation.ENTRY=original_entry
    controls=[]
    for job in jobs:
        if not job.get('original_state'):
            continue
        reference=Path(job['original_state']).with_suffix('.npz')
        if digest(reference)!=job['original_state_sha256']:
            raise ValueError('Original state arrays changed')
        with np.load(reference) as a,np.load(job['state_capture_target']+'.npz') as b:
            if set(a.files)!=set(b.files):
                raise ValueError('Passive capture changed state schema')
            for k in a.files:
                np.testing.assert_array_equal(a[k],b[k])
        controls.append(dict(case_id=job['recovery_case'],seed=job['seeds'][0],exact_original_state_replay=True))
    write(output/'completion.json',dict(status='complete',smoke=smoke,jobs=len(jobs),
        seconds=time.monotonic()-started,protocol_sha256=digest(output/'protocol.json'),replay_controls=controls,
        oral_used=False))
    print('Passive genealogy capture and retrospective smoothing complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path);parser.add_argument('--phase',choices=('benchmark','run'))
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output and args.phase=='benchmark':
        benchmark(args.config,args.output)
    elif args.config and args.output and args.phase=='run':
        run(args.config,args.output,args.smoke)
    else:
        parser.error('Provide --worker or --phase, --config and --output')
