"""Cross active capacity and belief retention at fixed SEARCH parameters.

Use --config CONFIG --output NEW_DIRECTORY; --smoke keeps one subject, two
saved path seeds and sixteen trials. All generation reuses the existing SEARCH
worker. This is an exploratory parameter intervention, not parameter fitting.
The unchanged point must exactly reproduce the archived SEARCH trajectories.
Sources may be completed fixed-parameter SEARCH studies or recalled-search
refit evaluations. Each source must identify one parameter point per subject.
``window_alignment: end`` anchors descriptive windows at the last trial;
the default ``start`` preserves previous output. The option controls process
summaries, with model trajectories and curve-scoring blocks preserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

from . import model_0923_recalled_search_refit as refit
from .model_0923_recalled_evidence_study import digest, read, write
from ...optimization.search.adaptive_proposals import candidate
from ...evaluation.model_0923_h_validation import block_paths, distribution_scores, jackknife_se
from ...evaluation.model_0923_h_generation_uncertainty import path_jackknife

ROOT = Path(__file__).resolve().parents[4]
MEMORY = 'engine.modules.memory_mod.kwargs.gamma'
CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'


def points(anchor: dict, memory: list[float], capacity: list[int]) -> list[dict]:
    """Change only the declared coordinates and retain the original point."""
    if (len(memory) < 2 or len(set(memory)) != len(memory)
            or any(isinstance(x, bool) or not np.isfinite(x) or not 0 <= x <= 1 for x in memory)
            or len(capacity) < 2 or len(set(capacity)) != len(capacity)
            or any(type(x) is not int or x < 2 for x in capacity)
            or anchor[MEMORY] not in memory or anchor[CAPACITY] not in capacity):
        raise ValueError('Require distinct valid levels including both original coordinates')
    return [candidate(dict(anchor, **{MEMORY: g, CAPACITY: c}), 'capacity_memory_intervention')
            for g, c in product(memory, capacity)]


def reference_jobs(source: Path, subject: int) -> list[Path]:
    """Accept one saved SEARCH bank and reject mixtures of parameter points."""
    legacy = sorted((source / 'jobs').glob(f'S{subject}_SEARCH_*.json'))
    fitted = sorted((source / 'jobs').glob(f'AUTO_evaluate_S{subject}_SEARCH_*.json'))
    if legacy and fitted:
        raise ValueError('Ambiguous SEARCH source layouts')
    files = legacy or fitted
    anchor, seeds = None, set()
    for file in files:
        job = read(file)
        point = job['candidate']['hyperparams']
        if (job.get('smoke') is not False or job['phase'] != 'autonomous' or job['model'] != 'SEARCH'
                or candidate(point, 'identity_check')['id'] != job['candidate']['id']):
            raise ValueError('Invalid autonomous SEARCH reference')
        if read(Path(job['folder']) / 'prepared.json')['subject'] != subject:
            raise ValueError('SEARCH source subject differs from the requested subject')
        if anchor is not None and point != anchor:
            raise ValueError('SEARCH source mixes parameter points')
        if len(set(job['seeds'])) != len(job['seeds']) or seeds.intersection(job['seeds']):
            raise ValueError('SEARCH source repeats path seeds')
        anchor = point
        seeds.update(job['seeds'])
    return files


def descriptive_windows(trials: int, width: int, alignment: str = 'start') -> list[tuple[int, int]]:
    """Partition all trials; place a partial window at the declared edge."""
    if trials < 1 or width < 1 or alignment not in ('start', 'end'):
        raise ValueError('Need positive window sizes and start/end alignment')
    if alignment == 'start':
        return [(start, min(start + width, trials)) for start in range(0, trials, width)]
    boundaries = sorted({0, *range(trials, 0, -width)})
    return list(zip(boundaries[:-1], boundaries[1:]))


def summarize(output: Path, jobs: list[dict], settings: dict, smoke: bool) -> None:
    groups, input_hashes, controls = {}, {}, 0
    for job in jobs:
        target = Path(job['target']); record = read(target.with_suffix('.json'))
        if digest(target.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Generated path hash changed')
        if any(pool['num_threads'] != 1 for pool in record['numeric_pools']):
            raise ValueError('Worker exceeded one numeric thread')
        with np.load(target.with_suffix('.npz')) as data:
            arrays = {key: data[key] for key in data.files}
        np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
        reference = Path(job['search_reference'])
        if digest(Path(job['search_original_job'])) != job['search_original_job_sha256']:
            raise ValueError('Original SEARCH job changed')
        if digest(reference) != job['search_reference_sha256']:
            raise ValueError('Original SEARCH reference changed')
        if job['anchor']:
            with np.load(reference) as previous:
                indices = [np.flatnonzero(previous['seeds'] == seed).item() for seed in job['seeds']]
                for key, value in arrays.items():
                    expected = (previous[key][indices] if key == 'seeds' else
                                previous[key][:record['trials']] if key == 'categories' else
                                previous[key][indices, :record['trials']])
                    np.testing.assert_array_equal(value, expected)
            controls += len(job['seeds'])
        key = (record['subject'], job['candidate']['id'])
        groups.setdefault(key, []).append((job, record, arrays))
        input_hashes[str(target.with_suffix('.npz').resolve())] = record['predictions_sha256']

    scores, path_rows, curve_rows, contrasts, banks = [], [], [], [], {}
    anchors = {}
    for (subject, point), parts in sorted(groups.items()):
        job, record, first = parts[0]
        categories = first['categories']
        for _, _, part in parts:
            np.testing.assert_array_equal(part['categories'], categories)
        arrays = {k: np.concatenate([a[k] for _, _, a in parts]) for k in first if k != 'categories'}
        order = np.argsort(arrays['seeds']); arrays = {k: v[order] for k, v in arrays.items()}
        n_paths = 2 if smoke else settings['paths']
        if len(arrays['seeds']) != n_paths or len(set(arrays['seeds'])) != n_paths:
            raise ValueError('Incomplete or duplicate path seed bank')
        if job['anchor']:
            anchors[subject] = point
        folder = Path(job['folder']); meta = read(folder / 'prepared.json')
        observed, human_categories = [], []
        for name in ('training.npz', 'sealed_test.npz'):
            if digest(folder / name) != meta['array_sha256'][name]:
                raise ValueError('Prepared observations changed')
            with np.load(folder / name) as human:
                observed.extend(human['choices'] == human['categories'])
                human_categories.extend(human['categories'])
        n_trials = len(categories); observed = np.asarray(observed[:n_trials])
        np.testing.assert_array_equal(categories, human_categories[:n_trials])
        correct = arrays['choices'] == categories[None, :]
        base = dict(subject=subject, point=point, anchor=job['anchor'], paths=n_paths,
                    memory=job['candidate']['hyperparams'][MEMORY],
                    capacity=job['candidate']['hyperparams'][CAPACITY])
        blocks = block_paths(correct, 0, n_trials, settings['block_width'])
        observed_blocks = block_paths(observed[None, :], 0, n_trials, settings['block_width'])[0]
        metric = distribution_scores(blocks, observed_blocks)
        errors = path_jackknife(blocks, observed_blocks)
        scores.append(dict(base, **metric, energy_mcse=jackknife_se(errors['energy'])))
        banks[subject, point] = dict(seeds=arrays['seeds'], energy=metric['energy'],
                                     jackknife=errors['energy'], arrays=arrays)
        for i in range(blocks.shape[1]):
            curve_rows.append(dict(base, trial=min((i+1)*settings['block_width'], n_trials),
                observed=float(observed_blocks[i]), mean=float(blocks[:, i].mean()),
                lower=float(np.quantile(blocks[:, i], .1)), upper=float(np.quantile(blocks[:, i], .9))))
        for start, stop in descriptive_windows(n_trials, settings['window'],
                                                settings.get('window_alignment', 'start')):
            for i, seed in enumerate(arrays['seeds']):
                present = arrays['present'][i, start:stop].astype(bool)
                path_rows.append(dict(base, seed=int(seed), start=start+1, stop=stop,
                    observed_accuracy=float(observed[start:stop].mean()),
                    accuracy=float(correct[i, start:stop].mean()),
                    correct_probability=float(arrays['correct_probability'][i, start:stop].mean()),
                    target_presence=float(present.mean()),
                    target_contribution=float(arrays['contribution'][i, start:stop].mean()),
                    target_beta_when_present=float(arrays['beta'][i, start:stop][present].mean()) if present.any() else np.nan,
                    active_rules=float(arrays['active_count'][i, start:stop].mean()),
                    search_events=int(arrays['search'][i, start:stop].sum()),
                    target_capacity_drops=int(arrays['target_capacity_dropped'][i, start:stop].sum()),
                    target_autonomous_exits=int(arrays['target_exited'][i, start:stop].sum())))
    for (subject, point), values in banks.items():
        baseline = banks[subject, anchors[subject]]
        np.testing.assert_array_equal(values['seeds'], baseline['seeds'])
        contrasts.append(dict(subject=subject, point=point, reference=anchors[subject],
            delta_energy=values['energy']-baseline['energy'],
            delta_energy_mcse=jackknife_se(values['jackknife']-baseline['jackknife'])))
    paths = pd.DataFrame(path_rows)
    keys = ['subject', 'point', 'anchor', 'paths', 'memory', 'capacity', 'start', 'stop']
    numeric = [x for x in paths.columns if x not in keys + ['seed']]
    windows = paths.groupby(keys, as_index=False)[numeric].mean()
    for name, frame in dict(scores=pd.DataFrame(scores), contrasts=pd.DataFrame(contrasts),
                            path_windows=paths, windows=windows, curves=pd.DataFrame(curve_rows)).items():
        frame.to_csv(output / f'{name}.csv', index=False)
    write(output / 'summary.json', dict(status='complete', smoke=smoke, exact_anchor_paths=controls,
        inputs=input_hashes, artifacts={p.name: digest(p) for p in output.glob('*.csv')},
        uncertainty='energy paired whole-path jackknife MCSE; descriptive process means; no refitted ranking',
        timing='H state and transition event at current choice, before its feedback'))


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new capacity-memory probe directory')
    settings = yaml.safe_load(config.read_text())
    if any(type(settings[k]) is not int or settings[k] < 1 for k in ('paths','window','block_width','parallel_budget')):
        raise ValueError('Paths, window sizes and worker budget must be positive integers')
    descriptive_windows(1, settings['window'], settings.get('window_alignment', 'start'))
    jobs, hashes, sources = [], {}, {}
    for subject, spec in list(settings['subjects'].items())[:1] if smoke else settings['subjects'].items():
        source = ROOT / spec['source']
        completion = read(source / 'completion.json')
        if completion['status'] != 'complete' or completion.get('smoke', False):
            raise ValueError('Require completed scientific SEARCH paths')
        sources[str(source)] = digest(source / 'completion.json')
        requested = 2 if smoke else settings['paths']
        left = requested
        for file in reference_jobs(source, subject):
            if left <= 0:
                break
            original = read(file); reference = Path(original['target']).with_suffix('.npz')
            seeds = original['seeds'][:left]; left -= len(seeds)
            for point in points(original['candidate']['hyperparams'], spec['memory'], spec['capacity']):
                name = f'S{subject}_{point["id"]}_{requested-left:03d}'
                job = deepcopy(original)
                job.update(name=name, candidate=point, seeds=seeds, smoke=smoke,
                    target=str((output / 'paths' / name).resolve()),
                    anchor=point['hyperparams']==original['candidate']['hyperparams'],
                    search_original_job=str(file.resolve()), search_original_job_sha256=digest(file),
                    search_reference=str(reference), search_reference_sha256=digest(reference))
                jobs.append(job); hashes.update(original['source_sha256'])
        if left:
            raise ValueError('Insufficient archived path seeds')
    for path in (Path(__file__), ROOT / refit.ENTRY,
                 ROOT / 'src/Bayesian_state/optimization/search/adaptive_proposals.py',
                 ROOT / 'src/Bayesian_state/evaluation/model_0923_h_validation.py',
                 ROOT / 'src/Bayesian_state/evaluation/model_0923_h_generation_uncertainty.py'):
        hashes[str(path.resolve().relative_to(ROOT))] = digest(path)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Original source changed: {name}')
    output.mkdir(parents=True); refit.base.archive(output, config, hashes)
    for job in jobs:
        job['source_sha256'] = hashes
    write(output / 'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        source_completion_sha256=sources, jobs=len(jobs), python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy','scipy','pandas','PyYAML')},
        scope='fixed SEARCH parameter intervention; declared subjects and coordinates; development',
        coordinates=[MEMORY,CAPACITY], data_use='no parameter fitting; autonomous own-choice feedback; oral excluded'))
    start = time.monotonic()
    refit.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    summarize(output, jobs, settings, smoke)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        paths=sum(len(j['seeds']) for j in jobs), seconds=time.monotonic()-start,
        summary_sha256=digest(output / 'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); run(args.config, args.output, args.smoke)
