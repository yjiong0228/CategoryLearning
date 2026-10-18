"""Compare last-active precision retention using existing autonomous engines.

Run --config CONFIG --output NEW_DIRECTORY; --smoke uses both declared points,
one saved seed per point, sixteen trials and one worker. Original path banks
are reused. One baseline path per point is rerun and must be bitwise identical.
All declared retained-precision paths use the original seeds and parameters.
The new mechanism applies to every rule, with no task-target input. Target
identity is used only for process diagnostics. Oral reports are not read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/reentry_precision.py'
DIAGNOSIS = 'src/Bayesian_state/workflows/analysis/model_0923_capacity_path_diagnosis.py'


def helpers():
    spec = importlib.util.spec_from_file_location('reentry_precision_refit', ROOT /
        'src/Bayesian_state/workflows/analysis/model_0923_recalled_search_refit.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    recalled = helpers(); v = recalled.base.validation(); original = recalled.refit.autonomous

    def generate(job, workflow):
        from src.Bayesian_state.model.modules.base_module import ModuleRole
        module = recalled.base.load('src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision', MECHANISM)
        cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        with module.reentry_precision(cls, enabled=job['retain_precision']) as events:
            original(job, workflow)
        target = Path(job['target']); record = v.read(target.with_suffix('.json'))
        with np.load(target.with_suffix('.npz')) as data:
            n = data['categories'].size; seeds = data['seeds'].copy()
            if not job['retain_precision']:
                with np.load(job['reference']) as baseline:
                    indices = [int(np.flatnonzero(baseline['seeds'] == seed).item()) for seed in seeds]
                    for key in data.files:
                        expected = (baseline[key][indices] if key == 'seeds' else baseline[key][:n]
                                    if key == 'categories' else baseline[key][indices, :n])
                        np.testing.assert_array_equal(data[key], expected, err_msg=f'Baseline differs: {key}')
        counts, target_counts = np.zeros((len(seeds), n), int), np.zeros((len(seeds), n), int)
        details = []
        if job['retain_precision']:
            if len(events) != len(seeds)*n:
                raise ValueError('Precision diagnostic omitted a path transition')
            for i, event in enumerate(events):
                path, trial = divmod(i, n)
                if event['trial'] != trial:
                    raise ValueError('Precision diagnostics changed trial order')
                rules, values = event['precision_restored_rules'], event['precision_restored_values']
                counts[path, trial] = len(rules)
                target_counts[path, trial] = int(record['target_rule'] in rules)
                for rule, beta in zip(rules, values):
                    details.append(dict(seed=int(seeds[path]), trial=trial+1, rule=rule, restored_precision=beta))
        with target.with_suffix('.precision.npz').open('xb') as stream:
            np.savez_compressed(stream, counts=counts, target_counts=target_counts, seeds=seeds)
        v.write(target.with_suffix('.precision.json'), dict(status='complete',
            retained_precision=job['retain_precision'], exact_baseline=not job['retain_precision'],
            restores=int(counts.sum()), target_restores=int(target_counts.sum()), events=details,
            arrays_sha256=v.digest(target.with_suffix('.precision.npz'))))

    recalled.refit.autonomous = generate
    recalled.worker(file)


def collect(parts: list[dict], *, baseline: bool) -> dict[str, np.ndarray]:
    chunks, categories = [], None
    for job in parts:
        file = Path(job['reference']) if baseline else Path(job['target']).with_suffix('.npz')
        with np.load(file) as z:
            indices = [int(np.flatnonzero(z['seeds'] == seed).item()) for seed in job['seeds']]
            n = 16 if job['smoke'] else len(z['categories'])
            if categories is not None: np.testing.assert_array_equal(z['categories'][:n], categories)
            categories = z['categories'][:n].copy()
            chunks.append({k:z[k][indices] if k == 'seeds' else z[k][indices, :n]
                           for k in z.files if k != 'categories'})
    arrays = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    order = np.argsort(arrays['seeds']); arrays = {k:v[order] for k,v in arrays.items()}
    if len(set(arrays['seeds'])) != len(arrays['seeds']):
        raise ValueError('Repeated autonomous seed')
    return dict(arrays, categories=categories)


def summarize(output: Path, jobs: list[dict], settings: dict, smoke: bool) -> None:
    from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths, distribution_scores, jackknife_se
    from src.Bayesian_state.evaluation.model_0923_h_generation_uncertainty import path_jackknife
    from src.Bayesian_state.workflows.analysis.model_0923_capacity_path_diagnosis import path_windows, summarize_paths
    v = helpers().base.validation(); table, scores, contrasts, curves = [], [], [], []
    hashes, exact_controls, restores = {}, 0, 0
    for job in jobs:
        target = Path(job['target']); record = v.read(target.with_suffix('.json'))
        precision = v.read(target.with_suffix('.precision.json'))
        if v.digest(target.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Autonomous output changed')
        if v.digest(target.with_suffix('.precision.npz')) != precision['arrays_sha256']:
            raise ValueError('Precision diagnostics changed')
        if any(pool['num_threads'] != 1 for pool in record['numeric_pools']):
            raise ValueError('Multiple numeric threads in worker')
        exact_controls += int(precision['exact_baseline']); restores += precision['restores']
        for suffix in ('.json', '.npz', '.precision.json', '.precision.npz'):
            path = target.with_suffix(suffix); hashes[str(path.resolve())] = v.digest(path)
    for point in settings['points']:
        parts = [j for j in jobs if j['candidate']['id'] == point and j['retain_precision']]
        if not parts: raise ValueError('A declared parameter point is missing')
        job = parts[0]; meta = v.read(Path(job['folder'])/'prepared.json'); observed = []
        for name in ('training.npz', 'sealed_test.npz'):
            file = Path(job['folder'])/name
            if v.digest(file) != meta['array_sha256'][name]: raise ValueError('Behavior input changed')
            hashes[str(file.resolve())] = v.digest(file)
            with np.load(file) as z: observed.extend(z['choices'] == z['categories'])
        summaries, energies = {}, {}
        for variant, baseline in [('RESET', True), ('RETAIN', False)]:
            data = collect(parts, baseline=baseline); n = len(data['categories'])
            if len(data['seeds']) != (1 if smoke else settings['paths']): raise ValueError('Incomplete path bank')
            correct = data['choices'] == data['categories'][None, :]
            blocks = block_paths(correct, 0, n, settings['block_width'])
            observed_blocks = block_paths(np.asarray(observed[:n])[None, :], 0, n, settings['block_width'])[0]
            energy = distribution_scores(blocks, observed_blocks)['energy']
            deletes = path_jackknife(blocks, observed_blocks)['energy'] if not smoke else np.full(1, np.nan)
            energies[variant] = (energy, deletes)
            base = dict(subject=meta['subject'], point=point, variant=variant, paths=len(data['seeds']),
                capacity=job['candidate']['hyperparams'][v.H_PATH+'max_capacity'])
            scores.append(dict(base, window='whole', metric='energy', value=energy,
                               mcse=jackknife_se(deletes) if not smoke else np.nan))
            for block in range(blocks.shape[1]):
                for row, seed in enumerate(data['seeds']):
                    curves.append(dict(base, seed=int(seed), trial=min((block+1)*settings['block_width'], n),
                        accuracy=float(blocks[row, block]), observed=float(observed_blocks[block])))
            frame = path_windows(data, settings['window'])
            for window, start, stop in [('first', 0, min(settings['window'], n)),
                                       ('late', max(0, n-settings['window']), n), ('whole', 0, n)]:
                rows = frame[frame.window == window].sort_values('seed')
                if not smoke:
                    values, errors = summarize_paths(rows)
                else:
                    values = {k:float(rows[k].iloc[0]) for k in ('presence','contribution','capacity_drops','autonomous_exits')}
                    errors = {k:np.full(1, np.nan) for k in values}
                values['accuracy'] = float(correct[:, start:stop].mean())
                accuracy = correct[:, start:stop].mean(axis=1)
                errors['accuracy'] = ((accuracy.sum()-accuracy)/(len(accuracy)-1)
                                      if not smoke else np.full(1, np.nan))
                summaries[variant, window] = (values, errors)
                for metric, value in values.items():
                    scores.append(dict(base, window=window, metric=metric, value=value,
                        mcse=jackknife_se(errors[metric]) if np.isfinite(errors[metric]).all() else np.nan))
            table.append(frame.assign(**base))
        delta = energies['RETAIN'][0]-energies['RESET'][0]
        error = energies['RETAIN'][1]-energies['RESET'][1]
        contrasts.append(dict(subject=meta['subject'], point=point, window='whole', metric='energy',
                               difference=delta, mcse=jackknife_se(error) if not smoke else np.nan))
        for window in ('first', 'late', 'whole'):
            a, ae = summaries['RESET', window]; b, be = summaries['RETAIN', window]
            for metric in a:
                error = be[metric]-ae[metric]
                contrasts.append(dict(subject=meta['subject'], point=point, window=window, metric=metric,
                    difference=b[metric]-a[metric], mcse=jackknife_se(error) if np.isfinite(error).all() else np.nan))
    for name, frame in [('path_windows',pd.concat(table,ignore_index=True)), ('scores',pd.DataFrame(scores)),
                        ('contrasts',pd.DataFrame(contrasts)), ('curves',pd.DataFrame(curves))]:
        frame.to_csv(output/f'{name}.csv',index=False)
    if exact_controls != len(settings['points']): raise ValueError('Incomplete exact baseline controls')
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,jobs=len(jobs),
        exact_baseline_paths=exact_controls,precision_restores=restores,
        input_sha256=hashes,artifacts={p.name:v.digest(p) for p in output.glob('*.csv')},
        uncertainty='paired whole-path jackknife MCSE at fixed parameters; no parameter or subject uncertainty'))


def run(config: Path, output: Path, smoke: bool) -> None:
    recalled = helpers(); v = recalled.base.validation(); settings = yaml.safe_load(config.read_text())
    if output.exists(): raise FileExistsError('Use a new precision-retention output directory')
    source = ROOT/settings['source']; completion = v.read(source/'completion.json')
    if completion['status'] != 'complete' or completion.get('smoke') is not False:
        raise ValueError('Require a completed scientific capacity probe')
    if len(set(settings['points'])) != len(settings['points']) or settings['paths'] < 3:
        raise ValueError('Require distinct points and at least three scientific paths')
    files = sorted((source/'jobs').glob('*.json')); groups = {point:[] for point in settings['points']}
    input_hashes = {str((source/'completion.json').resolve()):v.digest(source/'completion.json')}
    jobs, dependencies = [], {}
    for file in files:
        job = v.read(file); point = job['candidate']['id']
        if point not in groups: continue
        reference = Path(job['target']).with_suffix('.npz'); record = reference.with_suffix('.json')
        if v.digest(reference) != v.read(record)['predictions_sha256']: raise ValueError('Reference path changed')
        original = Path(job['search_original_job'])
        if v.digest(original) != job['search_original_job_sha256']: raise ValueError('Original SEARCH job changed')
        original_job = v.read(original)
        dependencies.update(original_job['source_sha256'])
        for path in (file, reference, record, original): input_hashes[str(path.resolve())] = v.digest(path)
        groups[point].append((job, reference))
    for point, parts in groups.items():
        if not parts or sum(len(j['seeds']) for j, _ in parts) != settings['paths']:
            raise ValueError('Point lacks the complete declared baseline bank')
        for i, (original, reference) in enumerate(parts[:1] if smoke else parts):
            job = deepcopy(original); name = f'RETAIN_{original["name"]}'
            job.update(name=name, seeds=original['seeds'][:1] if smoke else original['seeds'], smoke=smoke,
                reference=str(reference), reference_sha256=v.digest(reference), retain_precision=True,
                target=str((output/'paths'/name).resolve()))
            jobs.append(job)
            if i == 0:
                control = deepcopy(job); name = f'RESET_CONTROL_{original["name"]}'
                control.update(name=name,seeds=job['seeds'][:1],retain_precision=False,
                               target=str((output/'paths'/name).resolve()))
                jobs.append(control)
    for name in (ENTRY,MECHANISM,DIAGNOSIS,
                 'src/Bayesian_state/evaluation/model_0923_h_generation_uncertainty.py'):
        dependencies[name] = v.digest(ROOT/name)
    for name, sha in dependencies.items():
        if v.digest(ROOT/name) != sha: raise ValueError(f'Executing dependency changed: {name}')
    output.mkdir(parents=True); recalled.base.archive(output,config,dependencies)
    for job in jobs: job['source_sha256'] = dependencies
    v.write(output/'protocol.json',dict(settings=settings,smoke=smoke,jobs=len(jobs),
        source_sha256=dependencies,input_sha256=input_hashes,python=sys.version,
        numpy=np.__version__,pandas=pd.__version__,
        scope='All-rule last-active precision retention at fixed behavior-selected points; no fitting, no oral input',
        assumptions='Cache persists indefinitely; absent rules receive no learning; first-entry initialization unchanged',
        data_use='Own-choice autonomous feedback; original shared seeds and parameters; target used only in diagnostics'))
    # Reuse the established single-layer launcher; only the worker entry changes.
    recalled.ENTRY = ENTRY
    recalled.execute(jobs,output,1 if smoke else min(settings['parallel_budget'],len(os.sched_getaffinity(0))))
    for name, sha in input_hashes.items():
        if v.digest(Path(name)) != sha: raise ValueError('Probe input changed during execution')
    summarize(output,jobs,settings,smoke)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config,args.output,args.smoke)
    else: parser.error('Supply --config and --output, or --worker')
