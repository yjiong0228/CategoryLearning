"""Observe rule evidence on a fixed prefix of already generated path seeds.

Use --cross CROSS --output NEW_DIRECTORY [--smoke]. Full mode replays the first
declared four-seed autonomous chunk in every cross cell; smoke mode requires a
completed cross smoke fixture and replays its first one-seed chunk. Each saved
output array must match the original exactly. Observation wraps complete_trial
before feedback learning and records all rule evidence, exposure, contributions
and precision. No cognitive mechanism or parameter is changed. These twenty
full paths illustrate the process; the full 640-path bank establishes frequency.
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
CROSS = 'src/Bayesian_state/workflows/analysis/model_0923_online_parameter_cross.py'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def worker(file: Path) -> None:
    job = read(file)
    for path, sha in job['trace_inputs'].items():
        if digest(Path(path)) != sha: raise ValueError('Trace input changed')
    spec = importlib.util.spec_from_file_location('rule_use_cross', ROOT / CROSS)
    cross = importlib.util.module_from_spec(spec); spec.loader.exec_module(cross)
    online = cross.helpers(); retained = online.helpers(); recalled = retained.helpers()
    online.helpers = lambda: retained
    retained.helpers = lambda: recalled
    original_auto = recalled.refit.autonomous
    paths = []

    def with_trace(own_job, workflow):
        from src.Bayesian_state.model import StateModel
        from src.Bayesian_state.model.modules.base_module import ModuleRole
        original = StateModel.complete_trial
        identities = {}

        def observe(model, choice, feedback, **kwargs):
            identity = id(model)
            if identity not in identities:
                # Keep the object alive so independent paths cannot reuse its id.
                identities[identity] = dict(model=model, rows=[])
                paths.append(identities[identity]['rows'])
            h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            active = np.zeros(h.total_hypo, dtype=bool); active[h.active] = True
            identities[identity]['rows'].append(dict(active=active,
                evidence=h.retention_evidence.copy(), exposure=h.retention_exposure.copy(),
                base_prior=h.base_prior.copy(), contribution=h.execution_probabilities.copy(),
                beta=model._pending_trial.beta.copy(), dominant=int(h.dominant_hypothesis)))
            return original(model, choice, feedback, **kwargs)

        StateModel.complete_trial = observe
        try: original_auto(own_job, workflow)
        finally: StateModel.complete_trial = original

    recalled.refit.autonomous = with_trace
    (online.worker if job['trace_mechanism'] == 'ONLINE' else retained.worker)(file)
    target = Path(job['target']); record = read(target.with_suffix('.json'))
    elements = 0
    with np.load(job['reference']) as old, np.load(target.with_suffix('.npz')) as new:
        if set(old.files) != set(new.files): raise ValueError('Replay output fields changed')
        for key in old.files:
            np.testing.assert_array_equal(new[key], old[key]); elements += new[key].size
        target_rule = record['target_rule']
        arrays = {key:np.asarray([[row[key] for row in path] for path in paths]) for key in paths[0][0]}
        if arrays['active'].shape[:2] != new['choices'].shape:
            raise ValueError('Observer missed an autonomous trial')
        np.testing.assert_array_equal(arrays['active'][:,:,target_rule], new['present'])
        np.testing.assert_array_equal(arrays['contribution'][:,:,target_rule], new['contribution'])
        np.testing.assert_array_equal(np.where(arrays['active'][:,:,target_rule], arrays['beta'][:,:,target_rule], 0.), new['beta'])
        np.testing.assert_allclose(arrays['contribution'].sum(-1), 1., atol=1e-12, rtol=0)
        if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('Non-finite captured state')
        reconstructed = 0
        if job['trace_mechanism'] == 'ONLINE':
            inertia = job['candidate']['hyperparams']['engine.modules.hypo_transitions_mod.kwargs.execution_inertia']
            for path in range(len(paths)):
                for t in range(1, record['trials']):
                    active = arrays['active'][path,t]; weights = np.zeros(active.shape)
                    scores = arrays['evidence'][path,t,active]
                    weights[active] = arrays['base_prior'][path,t,active] * np.exp(scores - scores.max())
                    weights[arrays['dominant'][path,t-1]] *= np.exp(inertia)
                    weights /= weights.sum()
                    np.testing.assert_allclose(weights, arrays['contribution'][path,t], atol=1e-12, rtol=0)
                    reconstructed += 1
    for path, sha in job['trace_inputs'].items():
        if digest(Path(path)) != sha: raise ValueError('Trace input changed during replay')
    trace = target.with_suffix('.trace.npz')
    with trace.open('xb') as stream: np.savez_compressed(stream, **arrays, seeds=job['seeds'])
    write(target.with_suffix('.trace.json'), dict(status='complete', cell=job['trace_cell'],
        paths=len(paths), trials=record['trials'], target_rule=target_rule, exact_replay_elements=elements,
        online_contributions_reconstructed=reconstructed, source_sha256=digest(Path(__file__)),
        array_sha256=digest(trace), reference_sha256=job['trace_inputs'][job['reference']],
        selection='First declared autonomous chunk; no selection by outcome or oral report'))


def run(source: Path, output: Path, smoke: bool = False) -> None:
    if output.exists(): raise FileExistsError('Use a new trace destination')
    protocol = read(source / 'protocol.json')
    if protocol['smoke'] != smoke: raise ValueError('Use a source with the same smoke status')
    if smoke and read(source / 'completion.json')['status'] != 'complete':
        raise ValueError('Need the completed short comparison')
    inputs = {str((source / 'protocol.json').resolve()):digest(source / 'protocol.json'),
              str(Path(__file__).resolve()):digest(Path(__file__)), str((ROOT / CROSS).resolve()):digest(ROOT / CROSS)}
    jobs = []; reference_seeds = None
    for bank in protocol['banks']:
        actual = bank['roles']['autonomous'][0]['job']; target = Path(actual['target'])
        record = read(target.with_suffix('.json')); reference = target.with_suffix('.npz').resolve()
        if (record['predictions_sha256'] != digest(reference) or actual['smoke'] != smoke
                or actual['candidate']['id'] != bank['candidate']['id']
                or actual['candidate']['hyperparams'] != bank['candidate']['hyperparams']
                or len(actual['seeds']) != (1 if smoke else 4)):
            raise ValueError('Original trace reference differs from the declared cross cell')
        if reference_seeds is None: reference_seeds = actual['seeds']
        if reference_seeds != actual['seeds']: raise ValueError('Trace cells must share seeds')
        for file in (reference, target.with_suffix('.json'), target.with_suffix('.precision.json')):
            inputs[str(file.resolve())] = digest(file)
        job = dict(deepcopy(actual), trace_cell=bank['cell'], trace_mechanism=bank['mechanism'],
                   reference=str(reference), name=bank['cell'], target=str((output / 'paths' / bank['cell']).resolve()))
        if bank['mechanism'] == 'ONLINE':
            side = target.with_suffix('.online.json'); inputs[str(side.resolve())] = digest(side)
            # The earlier probe used the same mechanism before refit job tags existed.
            job.update(belief_intervention='online_retention_evidence', online_belief=True,
                       fixed_coordinates={'engine.modules.hypo_transitions_mod.kwargs.transfer_scale':.01})
        for name, sha in actual['source_sha256'].items():
            if digest(ROOT / name) != sha: raise ValueError('Cognitive source changed')
            inputs[str((ROOT / name).resolve())] = sha
        jobs.append(job)
    if len(jobs) != 5: raise ValueError('Trace every declared cross cell')
    output.mkdir(parents=True); (output / 'jobs').mkdir()
    for job in jobs: job['trace_inputs'] = dict(inputs)
    workers = min(1 if smoke else 5, len(os.sched_getaffinity(0)))
    write(output / 'protocol.json', dict(status='frozen_before_replay', smoke=smoke, inputs_sha256=inputs,
        jobs=jobs, workers=workers, numerical_threads=1,
        path_selection='First declared four seeds in each full cell; first seed of each smoke cell',
        original_parameters_changed=False, oral_used=False))

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='cross_rule_use_') as cache:
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE='1')
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                                '--worker', str(file.resolve())], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return read(Path(job['target']).with_suffix('.trace.json'))

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            records.append(future.result()); print(f'{len(records)}/5 exact traces complete', flush=True)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cells=5,
        paths=sum(r['paths'] for r in records), exact_replay_elements=sum(r['exact_replay_elements'] for r in records),
        online_contributions_reconstructed=sum(r['online_contributions_reconstructed'] for r in records),
        protocol_sha256=digest(output / 'protocol.json'), records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cross','output','worker'): parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.cross and args.output: run(args.cross,args.output,args.smoke)
    else: parser.error('Use --cross and --output, or an internal --worker')
