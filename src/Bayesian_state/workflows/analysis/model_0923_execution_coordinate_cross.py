"""Separate candidate evidence memory and search frequency in frozen executions.

Use --config YAML --output NEW [--smoke]. Complete the two-coordinate factorial
around an existing execution-by-parameter cross. Reuse all completed corner
cells, simulate the two hybrid parameter points under each execution policy,
and retain original whole-path seeds. This diagnostic does not select parameters.
Smoke executes one 16-trial path per cell with one worker and reproduces all
eight original cells exactly. Existing cognitive and subprocess workers are used.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import model_0923_execution_parameter_cross as execution
from .model_0923_h_validation import write
from ...evaluation import model_0923_h_validation as metrics
from ...evaluation.model_0923_execution_autonomous_review import combine_paths
from ...evaluation.model_0923_estimation_scope import validate_generation
from ...evaluation.model_0923_h_generation_uncertainty import path_jackknife
from ...evaluation.model_0923_separate_belief_refit import check_execution
from ...optimization.search.adaptive_proposals import candidate

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PREFIX = 'engine.modules.hypo_transitions_mod.kwargs.'
MEMORY = PREFIX + 'intervention.decay'
SEARCH = PREFIX + 'search_controller.baseline_probability'
ORIGINS = ('MIX', 'SELECT')


def hybrid_candidate(old: dict, new: dict, memory: str, search: str) -> dict:
    """Change only the two declared coordinates; reject additional confounding changes."""
    if memory not in ORIGINS or search not in ORIGINS:
        raise ValueError('Choose each coordinate from a frozen original point')
    a, b = old['hyperparams'], new['hyperparams']
    if set(a) != set(b) or {k for k in a if a[k] != b[k]} != {MEMORY, SEARCH}:
        raise ValueError('Require exactly evidence memory and search frequency differences')
    point = dict(a)
    point[MEMORY] = (a if memory == 'MIX' else b)[MEMORY]
    point[SEARCH] = (a if search == 'MIX' else b)[SEARCH]
    return candidate(point, 'frozen_execution_coordinate_cross',
                     parents=[old['id'], new['id']], memory_origin=memory, search_origin=search)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new coordinate cross directory')
    settings = yaml.safe_load(config.read_text())
    if set(settings) != {'reference', 'subject', 'parallel_budget'} or not 1 <= settings['parallel_budget'] <= 128:
        raise ValueError('Declare the reference cross, participant and available worker budget')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); sha = metrics.digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError('Input changed during the coordinate cross')
        inputs[str(path)] = sha
        return path

    def read(path):
        return metrics.load(bind(path))

    bind(config); bind(__file__)
    source = ROOT / settings['reference']; previous = read(source / 'protocol.json'); done = read(source / 'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or previous['smoke']
            or previous['settings']['subject'] != settings['subject']
            or metrics.digest(source / 'protocol.json') != done['protocol_sha256']):
        raise ValueError('Require the matching completed scientific cross')
    for path, sha in done['inputs_sha256'].items():
        if metrics.digest(bind(path)) != sha:
            raise ValueError('Verified reference input changed')
    original = {(c['model'], c['parameter'], c['policy']): c for c in previous['cells']}
    if len(original) != 8:
        raise ValueError('Require both proposal arms, parameter points and execution policies')
    reference_eval = read(ROOT / previous['settings']['original'] / 'protocol.json')
    fit = Path(reference_eval['fit']); frozen = read(fit / 'nominees.json')
    cells, jobs = [], []
    for arm, policy, memory, search in product(('POOL', 'SEARCH'), ORIGINS, ORIGINS, ORIGINS):
        old = original[arm, 'MIX', policy]['items']; new = original[arm, 'SELECT', policy]['items']
        point = hybrid_candidate(old[0]['job']['candidate'], new[0]['job']['candidate'], memory, search)
        existing = memory == search
        templates = original[arm, memory, policy]['items'] if existing else old
        if sum(len(item['job']['seeds']) for item in templates) != 128:
            raise ValueError('Require all original path seeds')
        cell = dict(model=arm, policy=policy, memory_origin=memory, search_origin=search,
                    candidate=point, items=[])
        for index, item in enumerate(templates[:1] if smoke else templates):
            template = item['job']; reused = existing and not smoke
            name = f'{arm}_{policy}_MEM_{memory}_SEARCH_{search}_C{index:03d}'
            if reused:
                job = template
            else:
                job = execution.cross_job(template, policy, output / 'chunks' / name, name, smoke)
                if not existing:
                    job['candidate'] = point
                job['source_sha256'][ENTRY] = metrics.digest(Path(__file__))
                jobs.append(job)
            if job['candidate']['hyperparams'] != point['hyperparams']:
                raise ValueError('Cell parameter assignment changed')
            cell['items'].append(dict(job=job, reused=reused,
                                      reference_target=template['target'] if existing else None))
        cells.append(cell)
    seed_banks = [[s for item in cell['items'] for s in item['job']['seeds']] for cell in cells]
    if any(bank != seed_banks[0] for bank in seed_banks) or len(set(seed_banks[0])) != len(seed_banks[0]):
        raise ValueError('Coordinate cells require identical unique seed banks')
    output.mkdir(parents=True); (output / 'chunks').mkdir()
    write(output / 'protocol.json', dict(coordinate_settings=settings, settings=frozen['settings'],
        fit=str(fit.resolve()), smoke=smoke,
        cells=cells, jobs=jobs, inputs_sha256=inputs, refitting=False, oral_used=False,
        scope='Development two-coordinate autonomous factorial at frozen points'))
    cross = execution.probe.helpers().helpers(); recalled = cross.helpers().helpers().helpers()
    recalled.ENTRY = execution.probe.ENTRY
    recalled.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    v = recalled.base.validation(); banks = {}; exact = 0
    for cell in cells:
        chunks = []
        for item in cell['items']:
            job = item['job']; target = Path(job['target'])
            if not item['reused'] and read(output / 'jobs' / (job['name'] + '.json')) != job:
                raise ValueError('Actual job differs from its declared cell')
            record, arrays = cross.check_chunk(job, 'SEPARATE', v, bind)
            if record['subject'] != settings['subject']:
                raise ValueError('Participant identity changed')
            check_execution(job, record, read(target.with_suffix('.belief.json')), read(target.with_suffix('.precision.json')))
            validate_generation(arrays, record)
            if not item['reused'] or cell['policy'] == 'SELECT' or target.with_suffix('.execution.json').exists():
                proof = read(target.with_suffix('.execution.json')); enabled = cell['policy'] == 'SELECT'
                if (proof['enabled'] != enabled or proof['policy'] != cell['policy']
                        or proof['updates'] != (len(job['seeds']) * record['trials'] if enabled else 0)
                        or not proof['uses_existing_dominant'] or proof['task_categories_used_for_selection']):
                    raise ValueError('Execution policy differs')
            if smoke and item['reference_target']:
                with np.load(bind(Path(item['reference_target']).with_suffix('.npz'))) as saved:
                    ix = [int(np.flatnonzero(saved['seeds'] == s).item()) for s in job['seeds']]
                    for key, value in arrays.items():
                        expected = saved[key][:record['trials']] if key == 'categories' else saved[key][ix] if key == 'seeds' else saved[key][ix, :record['trials']]
                        np.testing.assert_array_equal(value, expected); exact += value.size
            chunks.append(arrays)
        key = tuple(cell[k] for k in ('model', 'policy', 'memory_origin', 'search_origin'))
        banks[key] = combine_paths(chunks, seed_banks[0])
    if not smoke:
        score(output, cells, banks, read, bind)
    for path, sha in inputs.items():
        if metrics.digest(Path(path)) != sha:
            raise ValueError('Bound input changed during simulation')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cells=len(cells),
        jobs=len(jobs), new_paths=sum(len(j['seeds']) for j in jobs), exact_prefix_values=exact,
        inputs_sha256=inputs, protocol_sha256=metrics.digest(output / 'protocol.json'),
        artifacts_sha256={p.name: metrics.digest(p) for p in output.glob('*.csv')},
        uncertainty='Paired whole-path Monte Carlo errors; no parameter/population uncertainty'))


def score(output: Path, cells: list, banks: dict, read, bind) -> None:
    folder = Path(cells[0]['items'][0]['job']['folder']); meta = read(folder / 'prepared.json')
    observed = []
    for name in ('training.npz', 'sealed_test.npz'):
        with np.load(bind(folder / name)) as z:
            observed.append((z['choices'] == z['categories']).astype(float))
    y = np.concatenate(observed); cut = meta['training_trials']
    scores, paths, contrasts = [], [], []; estimates = {}
    windows = [('training', 0, cut), ('later', cut, len(y)), ('whole', 0, len(y)), ('last128', len(y)-128, len(y))]
    for key, a in banks.items():
        labels = dict(zip(('model', 'policy', 'memory_origin', 'search_origin'), key))
        correct = (a['choices'] == a['categories'][None, :]).astype(float)
        for window, first, last in windows:
            x = metrics.block_paths(correct, first, last); target = metrics.block_paths(y[None, :], first, last)[0]
            values = metrics.distribution_scores(x, target)
            estimates[*key, window] = values, path_jackknife(x, target)
            scores.append(dict(**labels, window=window, **values, realized_correct=correct[:, first:last].mean(),
                human_correct=y[first:last].mean(), target_present=a['present'][:, first:last].mean(),
                target_contribution=a['contribution'][:, first:last].mean()))
        for i, seed in enumerate(a['seeds']):
            found = np.flatnonzero(a['present'][i])
            paths.append(dict(**labels, seed=int(seed), ever_present=bool(len(found)),
                first_present_trial=int(found[0]+1) if len(found) else None,
                last128_correct=correct[i, -128:].mean(), last128_present=a['present'][i, -128:].mean(),
                last128_contribution=a['contribution'][i, -128:].mean(),
                autonomous_exits=int(a['target_exited'][i].sum()), capacity_drops=int(a['target_capacity_dropped'][i].sum()),
                search_events=int(a['search'][i].sum())))
    for arm, policy, fixed, window in product(('POOL', 'SEARCH'), ORIGINS, ORIGINS, [w[0] for w in windows]):
        for coordinate in ('memory', 'search'):
            akey = ('SELECT', fixed) if coordinate == 'memory' else (fixed, 'SELECT')
            bkey = ('MIX', fixed) if coordinate == 'memory' else (fixed, 'MIX')
            a, aj = estimates[arm, policy, *akey, window]; b, bj = estimates[arm, policy, *bkey, window]
            row = dict(model=arm, policy=policy, coordinate=coordinate, other_origin=fixed, window=window,
                       direction='SELECT_coordinate_minus_MIX_coordinate')
            for metric in ('energy', 'crps'):
                row['delta_' + metric] = a[metric] - b[metric]
                row['delta_' + metric + '_mc_se'] = metrics.jackknife_se(aj[metric] - bj[metric])
            contrasts.append(row)
    for name, rows in [('scores', scores), ('paths', paths), ('contrasts', contrasts)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    run(args.config, args.output, args.smoke)
