"""Cross frozen MIX/SELECT parameters and execution policies without refitting.

Use --config YAML --output NEW_DIRECTORY [--smoke]. A completed autonomous
review supplies verified original banks. Reuse its diagonal cells and simulate
only the opposite execution policies, preserving every seed and task input.
Smoke runs all eight cells with one original seed, 16 trials and one worker;
the four diagonal cells must reproduce their original arrays exactly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from ...evaluation import model_0923_h_validation as metrics
from ...evaluation.model_0923_execution_autonomous_review import combine_paths
from ...evaluation.model_0923_estimation_scope import validate_generation
from ...evaluation.model_0923_h_generation_uncertainty import path_jackknife
from ...evaluation.model_0923_separate_belief_refit import check_execution
from . import model_0923_dominant_execution_probe as probe
from .model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def cross_job(template: dict, policy: str, target: Path, name: str, smoke: bool) -> dict:
    """Change execution identity and output location, retaining the parameter point."""
    if policy not in ('MIX', 'SELECT') or template['phase'] != 'autonomous':
        raise ValueError('Require autonomous MIX/SELECT execution')
    job = deepcopy(template)
    job.update(name=name, target=str(target.resolve()), execution_policy=policy, smoke=smoke)
    if smoke:
        job['seeds'] = job['seeds'][:1]
    for relative in (ENTRY, probe.ENTRY, probe.EXECUTION):
        job['source_sha256'][relative] = metrics.digest(ROOT / relative)
    return job


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new execution/parameter cross directory')
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'review', 'original', 'revised', 'subject', 'parallel_budget'}
            or not 1 <= settings['parallel_budget'] <= 128):
        raise ValueError('Declare the completed review, two original evaluations and available workers')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); value = metrics.digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Input changed during the experiment')
        inputs[str(path)] = value
        return path

    def read(path):
        return metrics.load(bind(path))

    for file in (config, __file__, metrics.__file__, probe.__file__):
        bind(file)
    review = read(ROOT / settings['review'] / 'completion.json')
    if review['status'] != 'complete' or review['subject'] != settings['subject']:
        raise ValueError('Require the matching complete autonomous review')
    for file, value in review['inputs_sha256'].items():
        if metrics.digest(bind(file)) != value:
            raise ValueError('Reviewed input changed')
    cross = probe.helpers().helpers()
    v = cross.helpers().helpers().helpers().base.validation()
    cells, jobs = [], []
    for parameter, key in (('MIX', 'original'), ('SELECT', 'revised')):
        source = ROOT / settings[key]
        if str((source / 'protocol.json').resolve()) not in review['inputs_sha256']:
            raise ValueError('Evaluation is outside the verified review')
        protocol = read(source / 'protocol.json')
        for arm in ('POOL', 'SEARCH'):
            templates = [j for j in protocol['jobs'] if j['phase'] == 'autonomous' and j['model'] == arm]
            if sum(len(j['seeds']) for j in templates) != 128:
                raise ValueError('Require the original complete 128-path bank')
            for policy in ('MIX', 'SELECT'):
                cell = dict(parameter=parameter, policy=policy, model=arm, items=[])
                for index, template in enumerate(templates[:1] if smoke else templates):
                    original = read(source / 'jobs' / (template['name'] + '.json'))
                    if any(original[k] != value for k, value in template.items()):
                        raise ValueError('Executed template differs from its protocol')
                    for relative, value in original['source_sha256'].items():
                        if metrics.digest(bind(ROOT / relative)) != value:
                            raise ValueError('Original numerical implementation changed')
                    reused = policy == parameter and not smoke
                    name = f'{arm}_PARAM_{parameter}_EXEC_{policy}_C{index:03d}'
                    job = original if reused else cross_job(original, policy, output / 'chunks' / name, name, smoke)
                    if not reused:
                        jobs.append(job)
                    cell['items'].append(dict(job=job, reused=reused, original_target=original['target']))
                cells.append(cell)
    for arm in ('POOL', 'SEARCH'):
        banks = [[seed for item in c['items'] for seed in item['job']['seeds']] for c in cells if c['model'] == arm]
        if any(bank != banks[0] for bank in banks):
            raise ValueError('Crossed cells must preserve the complete common seeds')
    output.mkdir(parents=True); (output / 'jobs').mkdir(); (output / 'chunks').mkdir()
    workers = min(1 if smoke else settings['parallel_budget'], len(jobs), len(os.sched_getaffinity(0)))
    write(output / 'protocol.json', dict(settings=settings, smoke=smoke, cells=cells, jobs=jobs,
        inputs_sha256=inputs, workers=workers, numeric_threads=1, refitting=False, oral_used=False,
        scope='Development autonomous execution-by-frozen-parameter cross; original training nominees unchanged'))

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='execution_cross_') as cache:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / probe.ENTRY),
                                '--worker', str(file.resolve())], cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)

    print(f'{len(jobs)} autonomous jobs; {workers} workers; one numerical thread', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            future.result()
            if index % 8 == 0 or index == len(jobs):
                print(f'{index}/{len(jobs)} complete', flush=True)
    banks = {}; exact = 0
    for cell in cells:
        chunks = []
        for item in cell['items']:
            job = item['job']; target = Path(job['target'])
            row, arrays = cross.check_chunk(job, 'SEPARATE', v, bind)
            if row['subject'] != settings['subject']:
                raise ValueError('Autonomous subject differs')
            check_execution(job, row, read(target.with_suffix('.belief.json')), read(target.with_suffix('.precision.json')))
            validate_generation(arrays, row)
            if not item['reused'] or cell['policy'] == 'SELECT':
                execution = read(target.with_suffix('.execution.json'))
                enabled = cell['policy'] == 'SELECT'
                if (execution['policy'] != cell['policy'] or execution['enabled'] != enabled
                        or execution['updates'] != (len(job['seeds']) * row['trials'] if enabled else 0)
                        or not execution['uses_existing_dominant'] or execution['task_categories_used_for_selection']):
                    raise ValueError('Execution policy differs from its declared cell')
            if smoke and cell['policy'] == cell['parameter']:
                with np.load(bind(Path(item['original_target']).with_suffix('.npz'))) as z:
                    ix = [int(np.flatnonzero(z['seeds'] == seed).item()) for seed in job['seeds']]
                    for key, value in arrays.items():
                        expected = z[key][:row['trials']] if key == 'categories' else z[key][ix] if key == 'seeds' else z[key][ix, :row['trials']]
                        np.testing.assert_array_equal(value, expected); exact += value.size
            chunks.append(arrays)
        expected = [seed for item in cell['items'] for seed in item['job']['seeds']]
        banks[cell['model'], cell['parameter'], cell['policy']] = combine_paths(chunks, expected)
    scores, paths, contrasts = [], [], []; estimates = {}
    if not smoke:
        folder = Path(cells[0]['items'][0]['job']['folder']); meta = read(folder / 'prepared.json')
        observed = []
        for name in ('training.npz', 'sealed_test.npz'):
            with np.load(bind(folder / name)) as z:
                observed.append((z['choices'] == z['categories']).astype(float))
        y = np.concatenate(observed); cut = meta['training_trials']
        for (arm, parameter, policy), bank in banks.items():
            correct = (bank['choices'] == bank['categories'][None, :]).astype(float)
            for window, first, last in [('training', 0, cut), ('later', cut, len(y)), ('whole', 0, len(y)), ('last128', len(y)-128, len(y))]:
                samples = metrics.block_paths(correct, first, last); observed = metrics.block_paths(y[None, :], first, last)[0]
                values = metrics.distribution_scores(samples, observed)
                estimates[arm, parameter, policy, window] = (values, path_jackknife(samples, observed))
                scores.append(dict(model=arm, parameter=parameter, policy=policy, window=window, **values,
                    realized_correct=correct[:, first:last].mean(), target_present=bank['present'][:, first:last].mean(),
                    target_contribution=bank['contribution'][:, first:last].mean(), human_correct=y[first:last].mean()))
            for i, seed in enumerate(bank['seeds']):
                found = np.flatnonzero(bank['present'][i])
                paths.append(dict(model=arm, parameter=parameter, policy=policy, seed=int(seed),
                    ever_present=bool(len(found)), first_present_trial=int(found[0]+1) if len(found) else None,
                    last128_correct=correct[i, -128:].mean(), last128_present=bank['present'][i, -128:].mean(),
                    last128_contribution=bank['contribution'][i, -128:].mean(),
                    autonomous_exits=int(bank['target_exited'][i].sum()), capacity_drops=int(bank['target_capacity_dropped'][i].sum())))
        for arm in ('POOL', 'SEARCH'):
            for window in ('training', 'later', 'whole', 'last128'):
                comparisons = [(f'execution_at_{p}_parameters', (p, 'SELECT'), (p, 'MIX')) for p in ('MIX', 'SELECT')]
                comparisons += [(f'parameter_under_{p}_execution', ('SELECT', p), ('MIX', p)) for p in ('MIX', 'SELECT')]
                for name, target, reference in comparisons:
                    a, aj = estimates[arm, *target, window]; b, bj = estimates[arm, *reference, window]
                    row = dict(model=arm, window=window, comparison=name, direction='SELECT_minus_MIX')
                    for key in ('energy', 'crps'):
                        row['delta_' + key] = a[key] - b[key]
                        row['delta_' + key + '_mc_se'] = metrics.jackknife_se(aj[key] - bj[key])
                    contrasts.append(row)
        for name, rows in [('scores', scores), ('paths', paths), ('contrasts', contrasts)]:
            pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)
    for file, value in inputs.items():
        if metrics.digest(Path(file)) != value:
            raise ValueError('Bound input changed during simulation')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs), cells=len(cells),
        simulated_paths=sum(len(j['seeds']) for j in jobs), exact_diagonal_values=exact,
        protocol_sha256=metrics.digest(output / 'protocol.json'), inputs_sha256=inputs,
        artifacts_sha256={p.name:metrics.digest(p) for p in output.glob('*.csv')},
        uncertainty='Paired whole-path Monte Carlo error at fixed points; development diagnostic'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); run(args.config, args.output, args.smoke)
