"""Bounded fixed-parameter ONLINE versus separate-gamma belief experiment.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Reuse the declared first
autonomous chunks of all three ONLINE cross banks, then generate the same seeds
under separate belief memory. Both mechanisms receive identical new lower-cost
conditional filters. Choices select no new parameters; oral is excluded here.
The archived cognitive engine and existing numerical workers are reused.
Outputs include protocol, executed jobs, six aggregate banks, scores, paired
contrasts, path measures and source-bound completion records. Smoke uses one
worker, one path and one four-particle filter per parameter/mechanism.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
CROSS = 'src/Bayesian_state/workflows/analysis/model_0923_online_parameter_cross.py'
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/separate_belief_evidence.py'
INTERVENTION = 'separate_gamma_belief_evidence'
TRANSFER = 'engine.modules.hypo_transitions_mod.kwargs.transfer_scale'


def helpers():
    spec = importlib.util.spec_from_file_location('separate_belief_cross', ROOT / CROSS)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    job = json.loads(file.read_text())
    online = helpers().helpers()
    if job['probe_mechanism'] == 'ONLINE':
        online.worker(file)
        return
    if (job['probe_mechanism'] != 'SEPARATE' or job.get('online_belief')
            or job['belief_intervention'] != INTERVENTION):
        raise ValueError('Separate-belief job must declare its actual intervention')
    retained = online.helpers(); recalled = retained.helpers(); retained.helpers = lambda: recalled
    original_validation, original_auto = recalled.base.validation, recalled.refit.autonomous
    v = original_validation()

    def with_belief(function, own_job, workflow):
        module = recalled.base.load('src.Bayesian_state.model.modules.hypothesis_transition.separate_belief_evidence', MECHANISM)
        cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        with module.separate_belief_evidence(cls, enabled=True) as events:
            function(own_job, workflow)
        target = Path(own_job['target']); record = v.read(target.with_suffix('.json'))
        copies = len(own_job['seeds']) * (1 if own_job['phase'] == 'autonomous' else own_job['particles'])
        if Counter(e['trial'] for e in events) != {t: copies for t in range(1, record['trials'])}:
            raise ValueError('Separate evidence did not consume every completed feedback exactly once')
        expected_gamma = own_job['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma']
        if any(e['gamma'] != expected_gamma or not e['belief_matches_saved_state'] for e in events):
            raise ValueError('Separate evidence uses an incorrect memory or state')
        v.write(target.with_suffix('.belief.json'), dict(enabled=True, belief_intervention=INTERVENTION,
            gamma=expected_gamma, delta=sorted(set(e['delta'] for e in events)), updates=len(events),
            completed_feedback_timing_exact=True, belief_matches_saved_state=True,
            nonentry_updates=sum(e['newcomer_count'] == 0 for e in events),
            maximum_newcomer_episodes=max((e['newcomer_episodes'] for e in events), default=0),
            maximum_feedback_roundoff=max((e['feedback_roundoff_max'] for e in events), default=0.)))

    def validation():
        module = original_validation(); score = module.score
        module.score = lambda own_job, workflow: with_belief(score, own_job, workflow)
        return module

    recalled.base.validation = validation
    recalled.refit.autonomous = lambda own_job, workflow: with_belief(original_auto, own_job, workflow)
    retained.worker(file)


def prepare(config: Path, output: Path, smoke: bool):
    settings = yaml.safe_load(config.read_text())
    if (not 1 <= settings['parallel_budget'] <= 128 or settings['paths'] % 4
            or not 4 <= settings['paths'] <= 128 or not 2 <= settings['filter_seeds'] <= 8
            or not 4 <= settings['particles'] <= 512):
        raise ValueError('Declare bounded chunks, filter seeds, particles and worker allocation')
    cross = helpers(); v = cross.helpers().helpers().helpers().base.validation()
    source = ROOT / settings['smoke_cross' if smoke else 'cross']
    original = v.read(source / 'protocol.json')
    if original['smoke'] != smoke:
        raise ValueError('Source cross has a different smoke scope')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); sha = v.digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError('Input changed during preparation')
        inputs[str(path)] = sha
        return path

    for file in (config, Path(__file__), ROOT / MECHANISM, ROOT / CROSS, source / 'protocol.json'):
        bind(file)
    banks, jobs = [], []
    for bank in [b for b in original['banks'] if b['mechanism'] == 'ONLINE']:
        if bank['parameter'] not in ('RETAIN', 'ONLINE', 'NEAR'):
            raise ValueError('Unexpected parameter bank')
        for mechanism in ('ONLINE', 'SEPARATE'):
            own = dict(cell=f'{mechanism}_{bank["parameter"]}', mechanism=mechanism,
                       parameter=bank['parameter'], candidate=deepcopy(bank['candidate']), roles={})
            for role in ('autonomous', 'evaluate'):
                limit = 1 if smoke else (settings['paths'] // 4 if role == 'autonomous' else settings['filter_seeds'])
                items = bank['roles'][role][:limit]
                if len(items) != limit:
                    raise ValueError('Incomplete declared seed bank')
                own['roles'][role] = []
                for index, item in enumerate(items):
                    template = item['job']
                    if len(template['seeds']) != (1 if smoke or role == 'evaluate' else 4):
                        raise ValueError('Unexpected original chunk size')
                    for name, sha in template['source_sha256'].items():
                        if v.digest(bind(ROOT / name)) != sha:
                            raise ValueError('Original executing source changed')
                    if template['candidate'] != bank['candidate']:
                        raise ValueError('Original candidate changed')
                    reused = role == 'autonomous' and mechanism == 'ONLINE'
                    if reused:
                        cross.check_chunk(template, 'ONLINE', v, bind)
                        job = deepcopy(template)
                    else:
                        job = deepcopy(template)
                        job.update(name=f'{own["cell"]}_{role}_C{index:03d}',
                            target=str((output / 'chunks' / f'{own["cell"]}_{role}_C{index:03d}').resolve()),
                            probe_mechanism=mechanism, online_belief=mechanism == 'ONLINE',
                            belief_intervention='online_retention_evidence' if mechanism == 'ONLINE' else INTERVENTION,
                            fixed_coordinates={TRANSFER: .01})
                        job['source_sha256'].update({name: v.digest(ROOT / name) for name in (ENTRY, MECHANISM, CROSS)})
                        if role == 'evaluate': job['particles'] = 4 if smoke else settings['particles']
                        jobs.append(job)
                    own['roles'][role].append(dict(job=job, reused=reused))
            banks.append(own)
    if len(banks) != 6:
        raise ValueError('Need all three frozen ONLINE points and both mechanisms')
    return dict(settings=settings, original_settings=original['settings'], smoke=smoke,
        jobs=jobs, banks=banks, input_sha256=inputs, source_cross=str(source),
        workers=min(1 if smoke else settings['parallel_budget'], len(jobs), len(os.sched_getaffinity(0))),
        numerical_threads=1, scope='Fixed-parameter development screen; no parameter or oral selection',
        path_selection='First declared chunks, shared across mechanisms and parameter points',
        python=sys.version, versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pandas', 'PyYAML')}), cross, v


def summarize(output: Path, protocol: dict, rows: list[dict]) -> None:
    from src.Bayesian_state.evaluation.model_0923_reentry_refit import match_task, window_estimate
    from src.Bayesian_state.evaluation.model_0923_estimation_scope import paired_difference, validate_generation
    estimates, scores, paths, references = {}, [], [], []
    for row in rows:
        arrays = []
        for role in ('evaluate', 'autonomous'):
            with np.load(Path(row[role]['target']).with_suffix('.npz')) as z:
                arrays.append({k: z[k].copy() for k in z.files})
        c, a = arrays; meta = dict(row['evaluate']['record'], target_rule=row['autonomous']['record']['target_rule'])
        validate_generation(a, row['autonomous']['record'])
        current = c, a, meta
        if references: match_task(references[0], current)
        references.append(current)
        cut, length = meta['training_trials'], len(c['choices']); late = max(0, length - 128)
        for i, seed in enumerate(a['seeds']):
            paths.append(dict(cell=row['cell'], seed=int(seed),
                late_accuracy=float(np.mean(a['choices'][i, late:] == a['categories'][late:])),
                late_present=float(a['present'][i, late:].mean()),
                late_contribution=float(a['contribution'][i, late:].mean()),
                late_expected_error=float((1 - a['correct_probability'][i, late:]).mean()),
                target_exits=int(a['target_exited'][i].sum()),
                target_capacity_drops=int(a['target_capacity_dropped'][i].sum())))
        for phase, first, last in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
            estimate = window_estimate(c, a, first, last, protocol['original_settings']['features']['block_width'])
            estimates[row['cell'], phase] = estimate
            scores.append(dict(cell=row['cell'], phase=phase, **estimate['metrics']))
    contrasts = []
    for parameter in ('RETAIN', 'ONLINE', 'NEAR'):
        for phase in ('training', 'test', 'whole'):
            contrasts.append(dict(parameter=parameter, phase=phase,
                **paired_difference(estimates[f'SEPARATE_{parameter}', phase], estimates[f'ONLINE_{parameter}', phase])))
    pd.DataFrame(scores).to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False)
    pd.DataFrame(paths).to_csv(output / 'paths.csv', index=False)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists(): raise FileExistsError('Use a new separate-belief probe destination')
    protocol, cross, v = prepare(config, output, smoke)
    output.mkdir(parents=True); (output / 'jobs').mkdir()
    v.write(output / 'protocol.json', protocol)

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); v.write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='separate_belief_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                                '--worker', str(file.resolve())], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, job) for job in protocol['jobs']]), 1):
            future.result(); print(f'{i}/{len(protocol["jobs"])} probe jobs complete', flush=True)
    inputs = dict(protocol['input_sha256'])

    def bind(path):
        path = Path(path).resolve(); sha = v.digest(path)
        if str(path) in inputs and inputs[str(path)] != sha: raise ValueError('Bound probe input changed')
        inputs[str(path)] = sha
        return path

    rows = []
    for bank in protocol['banks']:
        row = {k: value for k, value in bank.items() if k != 'roles'}
        for role, items in bank['roles'].items():
            for item in items:
                job = item['job']
                record, _ = cross.check_chunk(job, 'ONLINE' if bank['mechanism'] == 'ONLINE' else 'RETAIN', v, bind)
                if bank['mechanism'] == 'SEPARATE':
                    proof = v.read(bind(Path(job['target']).with_suffix('.belief.json')))
                    copies = len(job['seeds']) * (1 if role == 'autonomous' else job['particles'])
                    if (proof['belief_intervention'] != INTERVENTION or proof['updates'] != copies * (record['trials'] - 1)
                            or not proof['completed_feedback_timing_exact'] or not proof['belief_matches_saved_state']):
                        raise ValueError('Missing separate-evidence execution proof')
            aggregate = deepcopy(items[0]['job']); target = output / 'banks' / f'{bank["cell"]}_{role}'
            target.parent.mkdir(exist_ok=True)
            aggregate.update(target=str(target.resolve()), seeds=[s for x in items for s in x['job']['seeds']])
            combined = v.combine_evaluation_chunks(aggregate, [x['job'] for x in items], role == 'autonomous')
            row[role] = dict(target=aggregate['target'], record=combined)
        rows.append(row)
    v.write(output / 'banks.json', dict(status='complete', rows=rows))
    if not smoke: summarize(output, protocol, rows)
    for name, sha in inputs.items():
        if v.digest(Path(name)) != sha: raise ValueError('Bound input changed during probe')
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, cells=len(rows),
        new_jobs=len(protocol['jobs']), inputs_sha256=inputs, protocol_sha256=v.digest(output / 'protocol.json'),
        artifacts={p.name: v.digest(p) for p in output.glob('*.csv')}, parameters_changed=False,
        scientific_scope='Bounded fixed-parameter screen; independent full validation and oral readout pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Use --config and --output, or an internal --worker')
