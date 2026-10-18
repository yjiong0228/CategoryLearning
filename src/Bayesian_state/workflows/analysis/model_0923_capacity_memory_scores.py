"""Score every capacity/retention intervention point on recorded choices.

Run --config CONFIG --output NEW_DIRECTORY. Reuse the original SEARCH nominee's
complete conditional filter bank and score every other grid point with the same
particles and filter seeds. The archived cognitive worker supplies all trial
predictions. Prefix, later and full losses use those same arrays. These are
retrospective development comparisons; oral content is excluded. --smoke keeps
the original particle count, two original seeds and eight trials, recomputes all
points, and checks the anchor exactly against the saved prediction prefixes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from . import model_0923_recalled_search_refit as refit
from .model_0923_recalled_capacity_memory_probe import CAPACITY, MEMORY, points
from .model_0923_estimation_scope import smoke_data
from ...evaluation.model_0923_h_validation import jackknife_scores, jackknife_se, nll
from ...optimization.search.adaptive_proposals import candidate, point_id

ROOT = refit.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
EXTRA = [ENTRY, 'src/Bayesian_state/workflows/analysis/model_0923_recalled_capacity_memory_probe.py',
         'src/Bayesian_state/workflows/analysis/model_0923_estimation_scope.py',
         'src/Bayesian_state/evaluation/model_0923_h_validation.py']


def grid_bank(jobs: list[dict], anchor: dict, memory: list[float], capacity: list[int]) -> list[dict]:
    """Match the completed intervention bank to the declared two coordinates."""
    expected = {row['id']: row for row in points(anchor['hyperparams'], memory, capacity)}
    seen = set()
    for job in jobs:
        row = job['candidate']
        if (row['id'] != point_id(row['hyperparams']) or row['id'] not in expected
                or row['hyperparams'] != expected[row['id']]['hyperparams']
                or job['anchor'] != (row['id'] == anchor['id'])):
            raise ValueError('Probe grid differs from the declared parameter intervention')
        seen.add(row['id'])
    if seen != set(expected):
        raise ValueError('Incomplete intervention grid')
    return [candidate(expected[key]['hyperparams'], 'completed_capacity_memory_intervention')
            for key in sorted(expected)]


def reference_seeds(jobs: list[dict], anchor: dict) -> list[int]:
    """Require a single numerical and cognitive specification across seeds."""
    if not jobs:
        raise ValueError('Missing original conditional jobs')
    same = ('particles', 'folder', 'runtime', 'mechanism', 'arm_settings', 'source_sha256')
    seeds = []
    for job in jobs:
        if (job['smoke'] or job['phase'] != 'evaluate' or job['model'] != 'SEARCH'
                or job['candidate']['id'] != anchor['id']
                or job['candidate']['hyperparams'] != anchor['hyperparams']
                or any(job[key] != jobs[0][key] for key in same)):
            raise ValueError('Conditional reference identity or settings differ')
        seeds.extend(job['seeds'])
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError('Need unique independent reference seeds')
    return seeds


def choice_contrasts(p: np.ndarray, reference: np.ndarray, cut: int) -> list[dict]:
    """Compare pooled probabilities using paired whole-filter jackknives."""
    p, reference = np.asarray(p, float), np.asarray(reference, float)
    if (p.ndim != 2 or p.shape != reference.shape or len(p) < 2 or not 0 < cut < p.shape[1]
            or any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in (p, reference))):
        raise ValueError('Need matching finite filter sequences and an internal time split')
    rows = []
    for phase, start, stop in [('prefix', 0, cut), ('later', cut, p.shape[1]), ('full', 0, p.shape[1])]:
        x, y = p[:, start:stop], reference[:, start:stop]
        jx, jy = jackknife_scores(x), jackknife_scores(y)
        rows.append(dict(phase=phase, trials=stop-start, choice_nll=nll(x),
            choice_nll_mc_se=jackknife_se(jx), delta_choice_nll=nll(x)-nll(y),
            delta_choice_nll_mc_se=jackknife_se(jx-jy)))
    return rows


def run(config: Path, output: Path, smoke: bool) -> None:
    v = refit.base.validation()
    settings = yaml.safe_load(config.read_text())
    probe, reference = ROOT/settings['probe'], ROOT/settings['reference']
    subject = settings['subject']
    if output.exists():
        raise FileExistsError('Use a new parameter-score directory')
    done, protocol = v.read(probe/'completion.json'), v.read(probe/'protocol.json')
    if done['status'] != 'complete' or done['smoke'] or protocol['smoke']:
        raise ValueError('Need completed scientific parameter interventions')
    if v.digest(probe/'summary.json') != done['summary_sha256']:
        raise ValueError('Parameter intervention summary changed')
    for path, expected in protocol['source_sha256'].items():
        if v.digest(probe/'source'/path) != expected:
            raise ValueError('Archived parameter intervention source changed')
    spec = protocol['settings']['subjects'][str(subject)]
    if (ROOT/spec['source']).resolve() != reference.resolve():
        raise ValueError('Probe and scoring refer to different original evaluations')
    original_protocol = v.read(reference/'protocol.json')
    original_fit = Path(original_protocol['fit'])
    frozen = v.read(original_fit/'nominees.json')
    if (v.read(reference/'completion.json')['status'] != 'complete' or frozen['smoke']
            or frozen['status'] != 'frozen_before_test'
            or original_protocol['nomination_sha256'] != v.digest(original_fit/'nominees.json')):
        raise ValueError('Incomplete or changed original evaluation')
    originals = [row for row in frozen['nominees'] if row['subject'] == subject and row['model'] == 'SEARCH']
    if len(originals) != 1:
        raise ValueError('Require one original SEARCH nominee')
    anchor = originals[0]['candidate']
    files = sorted((reference/'jobs').glob(f'evaluate_S{subject}_SEARCH_{anchor["id"]}_C*.json'))
    templates = [v.read(file) for file in files]
    seeds = reference_seeds(templates, anchor)
    if len(seeds) != settings['reference_seed_count'] or templates[0]['particles'] != settings['particles']:
        raise ValueError('Reference numerical budget differs from the declared budget')
    folder = Path(templates[0]['folder'])
    if v.read(folder/'prepared.json')['subject'] != subject:
        raise ValueError('Prepared subject identity changed')
    probe_files = sorted((probe/'jobs').glob(f'S{subject}_*.json'))
    bank = grid_bank([v.read(file) for file in probe_files], anchor, spec['memory'], spec['capacity'])
    summary = v.read(probe/'summary.json')
    inputs = {str(path.resolve()): v.digest(path) for path in [config, probe/'protocol.json',
        probe/'completion.json', probe/'summary.json', reference/'protocol.json',
        reference/'completion.json', original_fit/'nominees.json', *files, *probe_files,
        *[folder/name for name in ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz')]]}
    for name, expected in summary['artifacts'].items():
        if v.digest(probe/name) != expected:
            raise ValueError('Autonomous intervention table changed')
        inputs[str((probe/name).resolve())] = expected
    old_target = Path(templates[0]['target']).parent.parent / anchor['id']
    old_record = v.read(old_target.with_suffix('.json'))
    if (old_record['candidate']['id'] != anchor['id']
            or old_record['candidate']['hyperparams'] != anchor['hyperparams']
            or old_record['subject'] != subject or old_record['model'] != 'SEARCH'
            or old_record['particles'] != settings['particles'] or old_record['seeds'] != seeds):
        raise ValueError('Combined reference differs from the original parameter point or budget')
    if v.digest(old_target.with_suffix('.npz')) != old_record['predictions_sha256']:
        raise ValueError('Original combined conditional predictions changed')
    with np.load(old_target.with_suffix('.npz')) as data:
        np.testing.assert_array_equal(data['seeds'], seeds)
    for suffix in ('.npz', '.json'):
        path = old_target.with_suffix(suffix)
        inputs[str(path)] = v.digest(path)
    hashes = dict(templates[0]['source_sha256'])
    for name, expected in hashes.items():
        if v.digest(ROOT/name) != expected:
            raise ValueError(f'Original executing cognitive source changed: {name}')
        if protocol['source_sha256'].get(name) != expected:
            raise ValueError('Probe and likelihood calculation use different cognitive sources')
    hashes.update({name: v.digest(ROOT/name) for name in EXTRA})
    output.mkdir(parents=True)
    refit.base.archive(output, config, hashes)
    if smoke:
        folder = smoke_data(folder.parent, output, subject, v) / f'subject_{subject}'
        seeds = seeds[:2]
    jobs, plans = [], []
    for point in bank:
        reuse = point['id'] == anchor['id'] and not smoke
        target = old_target if reuse else output/'conditional'/point['id']
        job = dict(deepcopy(templates[0]), candidate=point, seeds=seeds, folder=str(folder.resolve()),
            name=f'evaluate_S{subject}_SEARCH_{point["id"]}', target=str(target.resolve()),
            smoke=smoke, source_sha256=hashes)
        chunks = []
        if not reuse:
            for index, seed in enumerate(seeds):
                chunk = dict(job, seeds=[seed], name=job['name']+f'_C{index:03d}',
                    target=str((target.parent/'chunks'/(target.name+f'_C{index:03d}')).resolve()))
                chunks.append(chunk)
                jobs.append(chunk)
        plans.append(dict(candidate=point, job=job, chunks=chunks, reused=reuse))
    v.write(output/'protocol.json', dict(settings=settings, smoke=smoke, probe=str(probe.resolve()),
        reference=str(reference.resolve()), source_sha256=hashes, input_sha256=inputs,
        plans=plans, particles=settings['particles'], seeds=seeds,
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy','pandas','scipy','PyYAML')},
        data_use='development retrospective choice comparison; complete choices used; oral excluded',
        parameters='only memory and capacity vary; same grid as completed autonomous intervention'))
    refit.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    records, arrays = [], {}
    for plan in plans:
        record = old_record if plan['reused'] else v.combine_evaluation_chunks(plan['job'], plan['chunks'], False)
        target = Path(plan['job']['target']).with_suffix('.npz')
        if v.digest(target) != record['predictions_sha256']:
            raise ValueError('Conditional output changed')
        with np.load(target) as data:
            z = {key:data[key] for key in data.files}
        np.testing.assert_array_equal(z['seeds'], seeds)
        if record['particles'] != settings['particles'] or record['subject'] != subject:
            raise ValueError('Numerical budget or subject changed')
        if not np.isfinite(z['probabilities']).all() or np.any(z['probabilities'] < 0):
            raise ValueError('Invalid category probabilities')
        np.testing.assert_allclose(z['probabilities'].sum(axis=-1), 1., atol=1e-12, rtol=0)
        np.testing.assert_array_equal(z['observed_probability'],
            z['probabilities'][:,np.arange(len(z['choices'])),z['choices']-1])
        arrays[plan['candidate']['id']] = z
        records.append(dict(candidate=plan['candidate'], target=str(target), reused=plan['reused'], record=record))
    baseline = arrays[anchor['id']]
    if smoke:
        with np.load(old_target.with_suffix('.npz')) as old:
            for key,value in baseline.items():
                expected = old[key][:2] if key == 'seeds' else old[key][:8] if key in ('choices','categories') else old[key][:2,:8]
                np.testing.assert_array_equal(value, expected)
    cut = v.read(folder/'prepared.json')['training_trials']
    rows, trials = [], []
    for row in records:
        point, z = row['candidate'], arrays[row['candidate']['id']]
        for key in ('choices','categories'):
            np.testing.assert_array_equal(z[key], baseline[key])
        identity = dict(subject=subject, model='SEARCH', point=point['id'], anchor=point['id']==anchor['id'],
            memory=point['hyperparams'][MEMORY], capacity=point['hyperparams'][CAPACITY],
            particles=settings['particles'], seeds=len(seeds))
        rows.extend(dict(identity, **score) for score in choice_contrasts(z['observed_probability'], baseline['observed_probability'], cut))
        pooled, original = z['observed_probability'].mean(axis=0), baseline['observed_probability'].mean(axis=0)
        loss, original_loss = [-np.log(np.maximum(x,np.finfo(float).tiny)) for x in (pooled,original)]
        for trial in range(len(pooled)):
            trials.append(dict(identity, trial=trial+1, phase='prefix' if trial < cut else 'later',
                observed_probability=pooled[trial], choice_nll=loss[trial], delta_choice_nll=loss[trial]-original_loss[trial],
                correct=int(z['choices'][trial]==z['categories'][trial])))
    pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
    pd.DataFrame(trials).to_csv(output/'trial_scores.csv', index=False)
    for name, expected in inputs.items():
        if v.digest(Path(name)) != expected:
            raise ValueError('Choice comparison input changed')
    for name, expected in hashes.items():
        if v.digest(ROOT/name) != expected:
            raise ValueError('Choice comparison source changed')
    v.write(output/'completion.json', dict(status='complete', smoke=smoke, subject=subject,
        points=len(bank), new_filter_jobs=len(jobs), reused_anchor=not smoke,
        exact_anchor_prefix_check=smoke, training_trials=cut, trials=len(baseline['choices']),
        records=records, artifacts_sha256={name:v.digest(output/name) for name in ('scores.csv','trial_scores.csv')}))
    print(f'{len(bank)} points scored with common particles and seeds; prefix/full contrasts saved',flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)


if __name__ == '__main__':
    main()
