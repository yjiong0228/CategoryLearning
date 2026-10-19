"""Replay frozen ONLINE/SEPARATE nominees to decompose autonomous errors.

Use --online-evaluation OLD --separate-evaluation NEW --output NEW_DIRECTORY
--workers N. Both complete autonomous components are required; full conditional
evaluation may still be running. Every original chunk and seed is retained.
--smoke checks one seed and sixteen trials from each arm/mechanism on one worker.
The existing archived cognitive workers run unchanged. Before feedback learning,
record rule evidence, belief, contributions and the target rule's emission.
Every original output array must replay exactly. Error allocation is descriptive
at the saved states; it is not an intervention or a new parameter selection.
Outputs: protocol, jobs, replay arrays, compact trace arrays and completion.
Oral reports are not read. Workers use one numeric thread; share the 128-process
allocation with other active work by supplying the available worker count.
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

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PROBE = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_probe.py'
CHANNELS = 'src/Bayesian_state/evaluation/model_0923_readout_channels.py'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    job = read(file)
    for path, value in job['trace_inputs'].items():
        if digest(Path(path)) != value:
            raise ValueError('Trace input changed')
    probe = load('refit_rule_use_probe', PROBE)
    cross = probe.helpers()
    online = cross.helpers()
    retained = online.helpers()
    recalled = retained.helpers()
    probe.helpers = lambda: cross
    cross.helpers = lambda: online
    online.helpers = lambda: retained
    retained.helpers = lambda: recalled
    original_auto = recalled.refit.autonomous
    traces = []
    with np.load(job['reference']) as source:
        categories = source['categories'].copy()
    target_rule = read(Path(job['audit']))['target']

    def observe_autonomous(own_job, workflow):
        from src.Bayesian_state.model import StateModel
        from src.Bayesian_state.model.modules.base_module import ModuleRole
        from src.Bayesian_state.model.modules.readout.choice import normalize_probability_vector

        original = StateModel.complete_trial
        identities = {}

        def observe(model, choice, feedback, **kwargs):
            identity = id(model)
            if identity not in identities:
                identities[identity] = dict(model=model, rows=[])
                traces.append(identities[identity]['rows'])
            rows = identities[identity]['rows']
            h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            pending = model._pending_trial
            active = np.asarray(h.active, dtype=int)
            if model.engine.get_module(ModuleRole.MAPPING) is not None:
                raise ValueError('This target-emission diagnostic requires fixed category labels')
            # The full mixture below is checked against the actually saved
            # probability, so a changed choice readout cannot silently pass.
            q = np.full(h.total_hypo, np.nan)
            for rule in active:
                raw = model.partition_model.get_category_probabilities(hypo=int(rule),
                    data=([pending.perceived_stimulus], [1], [1.]),
                    beta=float(pending.beta[rule]), distance_mode=model.engine.distance_mode)
                q[rule] = normalize_probability_vector(raw[:, 0], strict=True)[categories[pending.trial_index] - 1]
            evidence = h.belief_evidence if job['trace_mechanism'] == 'SEPARATE' else h.retention_evidence
            score = np.log(h.base_prior[active]) + evidence[active]
            belief = np.exp(score - score.max())
            belief /= belief.sum()
            present = target_rule in active
            index = int(np.flatnonzero(active == target_rule)[0]) if present else None
            before_inertia = np.zeros(h.total_hypo)
            before_inertia[active] = belief
            if rows:
                inertia = job['candidate']['hyperparams']['engine.modules.hypo_transitions_mod.kwargs.execution_inertia']
                weights = before_inertia.copy()
                weights[rows[-1]['dominant']] *= np.exp(inertia)
                weights /= weights.sum()
                np.testing.assert_allclose(weights, h.execution_probabilities, rtol=0, atol=1e-12)
            best_other = active[active != target_rule]
            other_scores = np.log(h.base_prior[best_other]) + evidence[best_other]
            if len(best_other):
                competitor = int(best_other[np.argmax(other_scores)])
                margin = float(score[index] - other_scores.max()) if present else np.nan
            else:
                competitor, margin = -1, np.nan
            rows.append(dict(present=int(present), target_correct=float(q[target_rule]) if present else np.nan,
                mixture_correct=float(np.dot(h.execution_probabilities[active], q[active])),
                contribution=float(h.execution_probabilities[target_rule]),
                target_beta=float(pending.beta[target_rule]) if present else 0.,
                target_belief=float(before_inertia[target_rule]),
                target_best=int(present and (not len(best_other) or margin >= 0)),
                target_margin=margin, competitor=competitor,
                competitor_correct=float(q[competitor]) if competitor >= 0 else np.nan,
                target_evidence=float(evidence[target_rule]) if present else np.nan,
                target_h_evidence=float(h.retention_evidence[target_rule]) if present else np.nan,
                active_count=len(active), dominant=int(h.dominant_hypothesis),
                previous_feedback=np.nan if not rows else float(rows[-1]['feedback']),
                feedback=float(feedback)))
            return original(model, choice, feedback, **kwargs)

        StateModel.complete_trial = observe
        try:
            original_auto(own_job, workflow)
        finally:
            StateModel.complete_trial = original

    recalled.refit.autonomous = observe_autonomous
    if job['trace_mechanism'] == 'ONLINE':
        online.worker(file)
    elif job['trace_mechanism'] == 'SEPARATE':
        probe.worker(file)
    else:
        raise ValueError('Expected declared ONLINE or SEPARATE mechanism')
    target = Path(job['target'])
    record = read(target.with_suffix('.json'))
    arrays = {key: np.asarray([[row[key] for row in path] for path in traces]) for key in traces[0][0]}
    checked = 0
    with np.load(job['reference']) as old, np.load(target.with_suffix('.npz')) as new:
        if set(old.files) != set(new.files):
            raise ValueError('Replay field set changed')
        indices = [int(np.flatnonzero(old['seeds'] == seed).item()) for seed in job['seeds']]
        n = record['trials']
        for key in new.files:
            reference = (old[key][:n] if key == 'categories' else old[key][indices]
                         if key == 'seeds' else old[key][indices, :n])
            np.testing.assert_array_equal(new[key], reference)
            checked += new[key].size
        for captured, saved in [('present', 'present'), ('contribution', 'contribution'),
                                ('target_beta', 'beta'), ('feedback', 'feedback'), ('active_count', 'active_count')]:
            np.testing.assert_array_equal(arrays[captured], new[saved])
            checked += arrays[captured].size
        np.testing.assert_allclose(arrays['mixture_correct'], new['correct_probability'], rtol=0, atol=1e-12)
        checked += arrays['mixture_correct'].size
        metrics = load('refit_rule_use_channels', CHANNELS)
        channels = metrics.error_channels(arrays['present'], arrays['contribution'], arrays['target_correct'],
                                         new['correct_probability'])
        arrays.update(channels, seeds=new['seeds'].copy(), categories=new['categories'].copy())
    for path, value in job['trace_inputs'].items():
        if digest(Path(path)) != value:
            raise ValueError('Trace input changed during replay')
    trace = target.with_suffix('.trace.npz')
    with trace.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write(target.with_suffix('.trace.json'), dict(status='complete', mechanism=job['trace_mechanism'],
        model=job['model'], candidate=job['candidate']['id'], paths=len(traces), trials=record['trials'],
        target_rule=target_rule, smoke=job['smoke'], exact_replay_elements=checked,
        contribution_reconstructions=len(traces)*(record['trials']-1),
        emission_mixture_maximum_error=float(np.max(np.abs(arrays['mixture_correct'] - (1-arrays['expected_error'])))),
        array_sha256=digest(trace), source_sha256=digest(Path(__file__))))


def run(online_evaluation: Path, separate_evaluation: Path, output: Path, workers: int, smoke: bool) -> None:
    from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import check_execution
    from src.Bayesian_state.workflows.analysis.model_0923_online_parameter_cross import check_chunk
    from types import SimpleNamespace

    if output.exists() or not 1 <= workers <= 128:
        raise ValueError('Use a new output directory and an available 1–128 worker allocation')
    inputs, jobs = {}, []

    def bind(path):
        path = Path(path).resolve()
        value = digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Input changed while preparing replay')
        inputs[str(path)] = value
        return path

    for relative in (ENTRY, PROBE, CHANNELS):
        bind(ROOT / relative)
    reference_seeds = {}
    for mechanism, source in [('ONLINE', online_evaluation), ('SEPARATE', separate_evaluation)]:
        protocol = read(bind(source / 'protocol.json'))
        frozen = read(bind(Path(protocol['fit']) / 'nominees.json'))
        if frozen['smoke'] or frozen['status'] != 'frozen_before_test':
            raise ValueError('Need frozen scientific nominees')
        if digest(Path(protocol['fit']) / 'nominees.json') != protocol['nomination_sha256']:
            raise ValueError('Evaluation nomination changed')
        declared = {j['target']: j for j in protocol['jobs'] if j['phase'] == 'autonomous'}
        actual = [(file, read(bind(file))) for file in sorted((source / 'jobs').glob('AUTO_*.json'))]
        if set(declared) != {j['target'] for _, j in actual}:
            raise ValueError('Incomplete declared autonomous jobs')
        for arm in ('POOL', 'SEARCH'):
            selected = [(file, j) for file, j in actual if j['model'] == arm]
            if smoke:
                selected = selected[:1]
            seeds = []
            nominee = next(n['candidate'] for n in frozen['nominees'] if n['subject'] == 208 and n['model'] == arm)
            for index, (file, original) in enumerate(selected):
                if original['smoke'] or original['candidate'] != nominee:
                    raise ValueError('Autonomous parameters differ from the frozen nominee')
                for key, value in declared[original['target']].items():
                    if original.get(key) != value:
                        raise ValueError('Executed autonomous job differs from its protocol')
                for path, value in original['source_sha256'].items():
                    if digest(bind(ROOT / path)) != value:
                        raise ValueError('Original executing source changed')
                row, _ = check_chunk(original, mechanism, SimpleNamespace(read=read, digest=digest), bind)
                if row['subject'] != 208 or row['condition'] != 2 or row['trials'] != 576:
                    raise ValueError('This declared replay covers subject 208, fixed-label four-category feedback')
                target = Path(original['target'])
                if mechanism == 'SEPARATE':
                    check_execution(original, row, read(bind(target.with_suffix('.belief.json'))),
                                    read(bind(target.with_suffix('.precision.json'))))
                job = deepcopy(original)
                name = f'{mechanism}_{arm}_C{index:03d}'
                job.update(name=name, trace_mechanism=mechanism, smoke=smoke,
                    reference=str(bind(target.with_suffix('.npz'))),
                    target=str((output / 'paths' / name).resolve()),
                    seeds=original['seeds'][:1] if smoke else original['seeds'])
                job['source_sha256'][ENTRY] = digest(ROOT / ENTRY)
                job['source_sha256'][CHANNELS] = digest(ROOT / CHANNELS)
                seeds.extend(job['seeds'])
                jobs.append(job)
            if len(set(seeds)) != len(seeds) or len(seeds) != (1 if smoke else 128):
                raise ValueError('Need every declared seed exactly once')
            if arm in reference_seeds:
                np.testing.assert_array_equal(seeds, reference_seeds[arm])
            reference_seeds[arm] = seeds
    # A worker binds only its own input and common source dependencies. Binding
    # the whole cohort in every worker would repeat large, unrelated reads.
    for job in jobs:
        job['trace_inputs'] = {job['reference']: inputs[job['reference']],
                              str((ROOT / ENTRY).resolve()): digest(ROOT / ENTRY),
                              str((ROOT / CHANNELS).resolve()): digest(ROOT / CHANNELS)}
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    (output / 'paths').mkdir()
    count = min(1 if smoke else workers, len(jobs), len(os.sched_getaffinity(0)))
    write(output / 'protocol.json', dict(status='frozen_before_replay', smoke=smoke, workers=count,
        numeric_threads=1, jobs=jobs, inputs_sha256=inputs, oral_used=False, nominees_changed=False,
        scope='All original autonomous seeds; descriptive error allocation at the original cognitive states',
        source_evaluations=[str(online_evaluation.resolve()), str(separate_evaluation.resolve())]))

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory(prefix='refit_rule_use_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                    '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, check=True)
        return read(Path(job['target']).with_suffix('.trace.json'))

    print(f'{len(jobs)} replay jobs; {count} workers; numeric threads=1', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 16 == 0 or len(records) == len(jobs):
                print(f'{len(records)}/{len(jobs)} exact replays complete', flush=True)
    for path, value in inputs.items():
        if digest(Path(path)) != value:
            raise ValueError('Bound original input changed')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(records),
        paths=sum(r['paths'] for r in records), exact_replay_elements=sum(r['exact_replay_elements'] for r in records),
        contribution_reconstructions=sum(r['contribution_reconstructions'] for r in records),
        emission_mixture_maximum_error=max(r['emission_mixture_maximum_error'] for r in records),
        protocol_sha256=digest(output / 'protocol.json'), inputs_sha256=inputs,
        trace_records_sha256={str(Path(j['target']).with_suffix('.trace.json')):
                              digest(Path(j['target']).with_suffix('.trace.json')) for j in jobs}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--online-evaluation', type=Path)
    parser.add_argument('--separate-evaluation', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.online_evaluation and args.separate_evaluation and args.output:
        run(args.online_evaluation, args.separate_evaluation, args.output, args.workers, args.smoke)
    else:
        parser.error('Supply both evaluations and a new --output')
