"""Compare rule-uniform and family-balanced discovery at frozen finite-H parameters.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Reuse the completed REHEARSE
nominees, task stimuli and autonomous seeds. RULE reproduces the original paths;
FAMILY changes only global candidate draws. Both save actual proposal, admission,
membership and response contributions. This is a fixed-parameter discovery
diagnostic; conditional state inference and refitting are separate evaluations.

Workers import the original fit's frozen core plus the new proposal adapter and
this recording entry. Smoke uses one subject, two seeds, eight training and four
later stimuli, one worker; the original eight-trial RULE prefix must be exact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load
from ...model.modules.base_module import ModuleRole
from ...model.modules.hypothesis_transition.family_balanced_proposal import balanced_global_proposal
from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from .model_0923_h_validation import write
from . import model_0923_finite_region_fit as original

ROOT = Path(__file__).resolve().parents[4]
MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_family_proposal_diagnostic'
ADAPTER = Path('src/Bayesian_state/model/modules/hypothesis_transition/family_balanced_proposal.py')


def worker(file: Path) -> None:
    job = load(file); original.verify(job['bindings'])
    if (job['stage'] != 'autonomous' or job['arm'] != 'REHEARSE'
            or job['proposal_arm'] not in ('RULE', 'FAMILY')
            or not Path(original.__file__).resolve().is_relative_to(Path(job['runtime']))):
        raise ValueError('Require the frozen REHEARSE runtime and a declared proposal')
    readout = original.read_choice_probabilities_from_model; rows = []

    def capture(model, stimulus):
        p, snapshot = readout(model, stimulus, return_snapshot=True)
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        ids = np.asarray(snapshot.hypothesis_indices, int); size = len(ids)
        event = h.transition_log[-1]
        row = dict(rule_ids=np.full(h.capacity, -1, dtype=np.int32),
            rule_weights=np.zeros(h.capacity), rule_prediction=np.zeros((h.capacity, 4)),
            beta=np.zeros(h.capacity), search=bool(event['search_event']),
            proposed=-1 if event['proposed_hypothesis'] is None else int(event['proposed_hypothesis']),
            admitted=bool(event['admission_event']), rejected=bool(event.get('newcomer_rejected', False)),
            capacity_dropped=-1 if event['capacity_dropped'] is None else int(event['capacity_dropped']))
        row['rule_ids'][:size] = ids; row['rule_weights'][:size] = snapshot.rule_weights
        row['rule_prediction'][:size] = snapshot.category_probabilities
        row['beta'][:size] = model.engine.beta[ids]
        np.testing.assert_array_equal(np.sort(ids), np.sort(h.active))
        np.testing.assert_allclose(snapshot.rule_weights @ snapshot.category_probabilities, p, atol=1e-12, rtol=0)
        rows.append(row)
        return p

    original.read_choice_probabilities_from_model = capture
    try:
        with balanced_global_proposal(original.Transition, enabled=job['proposal_arm'] == 'FAMILY'):
            original.worker(file)
    finally:
        original.read_choice_probabilities_from_model = readout
    target = Path(job['target']); traces = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    length = job['training_trials'] + job['test_trials']
    if len(rows) != length: raise ValueError('Incomplete prefeedback process trace')
    exact = 0
    if job['proposal_arm'] == 'RULE':
        with np.load(target.with_suffix('.npz')) as new, np.load(job['reference_arrays']) as old:
            for key in old.files:
                n = job['training_trials'] if job['smoke'] else length
                np.testing.assert_array_equal(new[key][:n], old[key][:n], err_msg=key)
                exact += new[key][:n].size
    trace_file = target.with_suffix('.process.npz')
    with trace_file.open('xb') as handle: np.savez_compressed(handle, **traces)
    original.verify(job['bindings'])
    write(target.with_suffix('.process.json'), dict(status='complete', proposal_arm=job['proposal_arm'],
        subject=job['subject'], seed=job['seed'], point=job['point'], trials=length,
        job_sha256=digest(file), arrays_sha256=digest(trace_file),
        behavior_sha256=digest(target.with_suffix('.npz')), original_elements_exact=exact))


def run(config: Path, output: Path, smoke: bool = False) -> None:
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'reference', 'paths', 'workers'} or type(settings['paths']) is not int
            or not 2 <= settings['paths'] <= 32 or type(settings['workers']) is not int
            or not 1 <= settings['workers'] <= 96):
        raise ValueError('Require a completed reference,2–32 paths and1–96 workers')
    output = output.resolve(); fit = (ROOT/settings['reference']).resolve()
    if output.exists(): raise FileExistsError('Use a new proposal diagnostic directory')
    done, pro, frozen = (load(fit/n) for n in ('completion.json', 'protocol.json', 'nominees.json'))
    if done['status'] != 'complete' or done['smoke'] or frozen['status'] != 'frozen_before_later_choices':
        raise ValueError('Require a complete full finite-H fit with frozen nominees')
    for name in ('protocol.json', 'nominees.json'):
        if digest(fit/name) != done['artifacts_sha256'][name]: raise ValueError('Reference metadata changed')
    subjects = pro['subjects'][:1] if smoke else pro['subjects']
    originals = []
    for s in subjects:
        seeds = pro['seeds'][str(s)]['autonomous'][:2 if smoke else settings['paths']]
        if len(seeds) != (2 if smoke else settings['paths']): raise ValueError('Reference seeds missing')
        for seed in seeds:
            matched = list((fit/'jobs').glob(f'autonomous_S{s}_REHEARSE_*_{seed}.json'))
            if len(matched) != 1: raise ValueError('Expected one original REHEARSE job')
            file = matched[0]; j = load(file); record = load(Path(j['target']).with_suffix('.json'))
            if (done['artifacts_sha256'][str(file.relative_to(fit))] != digest(file)
                    or record['job_sha256'] != digest(file)
                    or record['arrays_sha256'] != digest(Path(j['target']).with_suffix('.npz'))):
                raise ValueError('Reference path changed')
            original.load_trials(j); originals.append((file, j, record))
    output.mkdir(parents=True); (output/'jobs').mkdir(); (output/'chunks').mkdir()
    # Preserve the original cognitive implementation. Only this adapter and
    # recording entry are added to its archived source tree.
    shutil.copytree(fit/'source', output/'source')
    for source in (Path(__file__), ROOT/ADAPTER):
        target = output/'source'/source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): raise ValueError('New diagnostic would replace a frozen core file')
        shutil.copyfile(source, target)
    shutil.copyfile(config, output/'config.yaml')
    bindings = {str(config.resolve()): digest(config)}
    for p in (fit/'completion.json', fit/'protocol.json', fit/'nominees.json', Path(__file__), ROOT/ADAPTER):
        bindings[str(p)] = digest(p)
    for p in (output/'source').rglob('*.py'): bindings[str(p)] = digest(p)
    for file, j, record in originals:
        bindings.update(j['bindings'])
        for p in (file, Path(j['target']).with_suffix('.json'), Path(j['target']).with_suffix('.npz')):
            bindings[str(p)] = digest(p)
    original.verify(bindings)
    catalogue = [dict(rule=h.index, family=h.family) for h in build_relational_0923_space()]
    pd.DataFrame(catalogue).to_csv(output/'catalogue.csv', index=False)
    jobs = []
    for file, j, record in originals:
        for arm in ('RULE', 'FAMILY'):
            name = f'S{j["subject"]}_{arm}_{j["seed"]}'
            jobs.append(dict(deepcopy(j), name=name, proposal_arm=arm, smoke=smoke,
                target=str(output/'chunks'/name), reference_job=str(file),
                reference_arrays=str(Path(j['target']).with_suffix('.npz')), runtime=str(output/'source'),
                bindings=bindings, training_trials=8 if smoke else j['training_trials'],
                test_trials=4 if smoke else j['test_trials']))
    write(output/'protocol.json', dict(status='frozen_before_execution', settings=settings, smoke=smoke,
        bindings=bindings, subjects=subjects, catalogue=catalogue,
        source_fit=str(fit), nominees=[n for n in frozen['nominees'] if n['arm'] == 'REHEARSE'],
        stages='Autonomous process diagnostic only; parameters frozen from original training choices',
        scope='Change candidate distribution only; initial rule,search frequency,learning,admission,belief and execution fixed',
        oral_used=False, python=sys.version,
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    workers = min(1 if smoke else settings['workers'], len(jobs), len(os.sched_getaffinity(0)))
    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        launcher = ('import runpy,sys;sys.path.insert(0,sys.argv[1]);'
                    'sys.argv=sys.argv[2:];runpy.run_module(sys.argv[0],run_name="__main__")')
        with tempfile.TemporaryDirectory(prefix='family_proposal_') as cache, file.with_suffix('.log').open('x') as log:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-c', launcher, str(output/'source'), MODULE, '--worker', str(file)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return load(Path(job['target']).with_suffix('.process.json'))
    print(f'{len(jobs)} autonomous jobs, {workers} single-thread workers', flush=True); records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            records.append(future.result())
            if i % 8 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)
    original.verify(bindings)
    write(output/'completion.json', dict(status='complete', smoke=smoke, paths=len(jobs),
        original_elements_exact=sum(r['original_elements_exact'] for r in records),
        protocol_sha256=digest(output/'protocol.json'),
        artifacts={str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                   if p.is_file() and 'source' not in p.relative_to(output).parts and p.suffix != '.log'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Supply --worker or both --config and --output')
