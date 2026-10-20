"""Behavior-only parameter screening for finite-H region associations.

Run --config YAML --output NEW_DIRECTORY [--smoke]. FROZEN and REHEARSE
receive identical parameter grids, filter seeds and budgets. Training choices
select a fixed number of finalists including the declared anchor. Independent,
larger filters confirm the nominees before later choices are opened. Evaluation
continues learning from feedback at frozen parameters; autonomous paths generate
their own choices and feedback. All model algorithms use the shared core.

The grid estimates association rate, initial precision and two evidence retention
rates. Capacity, search, precision learning rate and physical perception remain
at the source diagnostic settings. This is a bounded development comparison.
Oral data are absent from every stage. Workers run an archived source snapshot.
Smoke uses one subject, eight training trials, two particles and one worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load, nll, jackknife_scores, jackknife_se
from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...metrics.task import category_learning_feedback
from ...model import ModelContext, StateModel
from ...model.modules.hypothesis_transition.finite_region_learning import finite_region_learning
from ...model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from ...model.modules.readout.choice import read_choice_probabilities_from_model
from ...utils.seeding import stable_seed
from .model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[4]
MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_finite_region_fit'
COORDINATES = ('association_learning_rate', 'beta_init', 'delta', 'gamma')
ARMS = ('FROZEN', 'REHEARSE')


def parameter_points(grid: dict) -> list[dict]:
    """Enumerate the declared common grid with stable, collision-checked IDs."""
    if set(grid) != set(COORDINATES):
        raise ValueError('Declare exactly the four supported coordinates')
    for key, values in grid.items():
        if (not values or len(set(values)) != len(values)
                or any(isinstance(v, bool) or not np.isfinite(v) for v in values)):
            raise ValueError('Require distinct finite grid values')
        if key == 'association_learning_rate':
            valid = all(v > 0. for v in values)
        elif key == 'beta_init':
            valid = all(.1 <= v <= 25. for v in values)
        else:
            valid = all(0. <= v <= 1. for v in values)
        if not valid:
            raise ValueError('Unsupported parameter range: ' + key)
    points = []
    for values in itertools.product(*(grid[k] for k in COORDINATES)):
        params = dict(zip(COORDINATES, map(float, values)))
        identity = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        points.append(dict(id=identity, parameters=params))
    if len({p['id'] for p in points}) != len(points):
        raise ValueError('Parameter identity collision')
    return points


def finalist_ids(rows: list[dict], anchor: str, count: int) -> list[str]:
    """Keep exactly count points: the anchor and the best other training scores."""
    if (len({r['id'] for r in rows}) != len(rows) or anchor not in {r['id'] for r in rows}
            or not 1 <= count <= len(rows) or any(not np.isfinite(r['nll']) for r in rows)):
        raise ValueError('Require unique finite training scores and an included anchor')
    ordered = sorted((r for r in rows if r['id'] != anchor), key=lambda r: (r['nll'], r['id']))
    return [anchor] + [r['id'] for r in ordered[:count-1]]


def configured_engine(template: dict, point: dict) -> dict:
    """Change the declared coordinates; association rate belongs to the adapter."""
    cfg = deepcopy(template)
    p = point['parameters']
    cfg['modules']['beta_mod']['kwargs']['beta_init'] = p['beta_init']
    cfg['modules']['memory_mod']['kwargs']['gamma'] = p['gamma']
    cfg['modules']['hypo_transitions_mod']['kwargs']['intervention']['decay'] = p['delta']
    cfg['oral_readout'] = None
    cfg['provenance']['label_mapping'] = {'mode': 'independent_region_response_rows'}
    cfg['provenance']['scientific_status'] = 'finite_H_bounded_behavioral_estimation'
    return cfg


def verify(bindings: dict) -> None:
    for filename, expected in bindings.items():
        if digest(Path(filename)) != expected:
            raise ValueError('Bound input changed: ' + filename)


def load_trials(job: dict) -> dict[str, np.ndarray]:
    """Training jobs cannot open later choices; evaluation requires a frozen nominee."""
    phase = job['stage']
    if phase not in ('screen', 'confirmation', 'evaluation', 'autonomous'):
        raise ValueError('Unknown estimation stage')
    with np.load(job['training']) as saved:
        data = {k: saved[k][:job['training_trials']] for k in ('stimulus', 'choices', 'feedback', 'categories')}
    if phase in ('screen', 'confirmation'):
        if 'later' in job or 'nominees' in job:
            raise ValueError('Training job contains evaluation inputs')
    else:
        frozen = load(Path(job['nominees']))
        if (digest(Path(job['nominees'])) != job['nomination_sha256']
                or frozen['status'] != 'frozen_before_later_choices'
                or not any(n['subject'] == job['subject'] and n['arm'] == job['arm']
                           and n['point'] == job['point'] for n in frozen['nominees'])):
            raise ValueError('Evaluation differs from the frozen nominee')
        with np.load(job['later']) as saved:
            data = {k: np.concatenate([v, saved[k][:job['test_trials']]]) for k, v in data.items()}
    x, y, f, c = (data[k] for k in ('stimulus', 'choices', 'feedback', 'categories'))
    expected = job['training_trials'] + (job['test_trials'] if phase in ('evaluation', 'autonomous') else 0)
    if (x.shape != (expected, 4) or not np.isfinite(x).all() or np.any((x < 0.) | (x > 1.))
            or any(v.shape != (expected,) for v in (y, f, c))
            or not np.isin(y, [1, 2, 3, 4]).all() or not np.isin(c, [1, 2, 3, 4]).all()
            or not np.array_equal(f, (y == c).astype(float))):
        raise ValueError('Invalid or misaligned condition-2 trial arrays')
    return data


def worker(file: Path) -> None:
    job = load(file)
    verify(job['bindings'])
    data = load_trials(job)
    cfg = configured_engine(job['engine'], job['point'])
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='finite_region_fit_') as cache:
        cfg['partition']['kwargs']['similarity_cache_dir'] = cache
        with finite_region_learning(Transition, learning_rate=job['point']['parameters']['association_learning_rate'],
                                    rehearse_candidates=job['arm'] == 'REHEARSE'):
            if job['stage'] == 'autonomous':
                cfg['modules']['hypo_transitions_mod']['kwargs']['module_seed'] = stable_seed(
                    dict(seed=job['seed'], role='finite_H_transition'))
                cfg['modules']['perception_mod']['kwargs']['module_seed'] = stable_seed(
                    dict(seed=job['seed'], role='finite_H_perception'))
                rng = np.random.default_rng(stable_seed(dict(seed=job['seed'], role='finite_H_choice')))
                model = StateModel(cfg, context=ModelContext(condition=2))
                predictions, responses, outcomes = [], [], []
                for x, category in zip(data['stimulus'], data['categories']):
                    prepared = model.begin_trial(x)
                    p = read_choice_probabilities_from_model(model, prepared.perceived_stimulus)
                    y = int(rng.choice(4, p=p) + 1)
                    f = category_learning_feedback(y, int(category), condition=2)
                    predictions.append(p); responses.append(y); outcomes.append(f)
                    model.complete_trial(y, f)
                arrays = dict(probabilities=np.asarray(predictions), generated_choices=np.asarray(responses),
                              generated_feedback=np.asarray(outcomes), categories=data['categories'])
            else:
                result = run_state_model_particle_filter(engine_config=cfg, subject_id=job['subject'],
                    condition=2, stimulus=data['stimulus'], choices=data['choices'], feedback=data['feedback'],
                    particle_count=job['particles'], particle_blocks=1, choice_readout_power=1.,
                    filter_seed=job['seed'], resample_threshold_fraction=.5)
                p = result.marginal_probabilities
                arrays = dict(probabilities=p, observed_probability=p[np.arange(len(p)), data['choices']-1],
                              choices=data['choices'], feedback=data['feedback'],
                              final_weights=result.final_weights, resampled=result.resampled)
    p = arrays['probabilities']
    if not np.isfinite(p).all() or np.any(p < 0.) or p.shape != (len(data['choices']), 4):
        raise ValueError('Invalid behavioral predictions')
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(p[0], .25, atol=1e-12, rtol=0)
    verify(job['bindings'])
    target = Path(job['target'])
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    write(target.with_suffix('.json'), dict(status='complete', job_sha256=digest(file),
        arrays_sha256=digest(target.with_suffix('.npz')), stage=job['stage'], subject=job['subject'],
        arm=job['arm'], point=job['point'], seed=job['seed'], particles=job['particles'],
        trials=len(p), source_file=str(Path(__file__).resolve()), elapsed_seconds=time.monotonic()-start))


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new finite-H fit directory')
    settings = yaml.safe_load(config.read_text())
    if (set(settings['subjects']) != {207, 208} or not 1 <= settings['workers'] <= 96
            or not 2 <= settings['autonomous_paths'] <= 128):
        raise ValueError('Require the declared two-subject bounded development comparison')
    for stage in ('screen', 'confirmation', 'evaluation'):
        if not 2 <= settings[stage]['particles'] <= 512 or not 2 <= settings[stage]['repeats'] <= 8:
            raise ValueError('Require bounded independent filters')
    if not (settings['screen']['particles'] < settings['confirmation']['particles']
            <= settings['evaluation']['particles']):
        raise ValueError('Confirmation and evaluation must increase particle resolution')
    points = parameter_points(settings['grid'])
    anchor = next((p for p in points if p['parameters'] == settings['anchor']), None)
    if anchor is None or not 2 <= settings['finalists'] <= len(points):
        raise ValueError('Require the declared anchor and at least two finalists')
    reference = ROOT/settings['reference']
    original = load(reference/'protocol.json'); done = load(reference/'completion.json')
    if done['status'] != 'complete' or done['protocol_sha256'] != digest(reference/'protocol.json'):
        raise ValueError('Require a complete source diagnostic')
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    selected_points = [anchor] if smoke else points
    templates, folders, metadata = {}, {}, {}
    inputs = {str(p.resolve()): digest(p) for p in (config, reference/'protocol.json', reference/'completion.json')}
    for subject in subjects:
        source = next(j for j in original['jobs'] if j['subject'] == subject)
        templates[subject] = source['engine']
        folder = Path(source['data_files'][0]).parent
        meta = load(folder/'prepared.json')
        if meta['subject'] != subject or meta['condition'] != 2:
            raise ValueError('Subject or condition changed')
        if digest(folder/'training.npz') != meta['array_sha256']['training.npz']:
            raise ValueError('Training data changed')
        folders[subject], metadata[subject] = folder, meta
        for path in (folder/'prepared.json', folder/'training.npz'):
            inputs[str(path)] = digest(path)
    output = output.resolve()
    output.mkdir(parents=True)
    for name in ('jobs', 'chunks', 'source'):
        (output/name).mkdir()
    shutil.copyfile(config, output/'config.yaml')
    # Every worker imports this immutable snapshot. Ongoing unrelated jobs can
    # continue with their existing source bindings in the main worktree.
    sources = [ROOT/'src/__init__.py'] + list((ROOT/'src/Bayesian_state').rglob('*.py'))
    for path in sources:
        target = output/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inputs[str(target)] = digest(target)
    seeds = {}
    for subject in subjects:
        seeds[subject] = {stage: [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
            subject=subject, stage=stage, repeat=i)) for i in range(
                1 if smoke else settings['autonomous_paths'] if stage == 'autonomous' else settings[stage]['repeats'])]
            for stage in ('screen', 'confirmation', 'evaluation', 'autonomous')}
    all_seeds = [s for banks in seeds.values() for bank in banks.values() for s in bank]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('Stage or subject seed collision')
    workers = min(1 if smoke else settings['workers'], len(os.sched_getaffinity(0)))
    write(output/'protocol.json', dict(status='frozen_before_execution', settings=settings, smoke=smoke,
        subjects=subjects, points=selected_points, seeds=seeds, anchor=anchor, templates=templates,
        inputs_sha256=inputs, workers=workers, numeric_threads=1,
        objective='mean per-trial negative log of independent-filter mean predictive choice probability',
        selection_data='training choices only', oral_used=False, source_runtime=str(output/'source'),
        scope='Four-coordinate development fit; capacity/search/precision-rate/physical perception fixed',
        python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')}))

    def make_job(subject, arm, point, stage, seed):
        name = f'{stage}_S{subject}_{arm}_{point["id"]}_{seed}'
        job = dict(name=name, stage=stage, subject=subject, arm=arm, point=point, seed=seed,
            engine=templates[subject], training=str(folders[subject]/'training.npz'),
            training_trials=8 if smoke else metadata[subject]['training_trials'],
            particles=1 if stage == 'autonomous' else 2 if smoke else settings[stage]['particles'],
            target=str(output/'chunks'/name), bindings=inputs.copy())
        if stage in ('evaluation', 'autonomous'):
            later = folders[subject]/'sealed_test.npz'
            if digest(later) != metadata[subject]['array_sha256']['sealed_test.npz']:
                raise ValueError('Later data changed')
            job.update(later=str(later), test_trials=4 if smoke else metadata[subject]['test_trials'],
                       nominees=str(output/'nominees.json'), nomination_sha256=digest(output/'nominees.json'))
            job['bindings'].update({str(later): digest(later), str(output/'nominees.json'): job['nomination_sha256']})
        return job

    def execute(jobs):
        def launch(job):
            file = output/'jobs'/(job['name']+'.json'); write(file, job)
            launcher = ('import runpy,sys;sys.path.insert(0,sys.argv[1]);'
                        'sys.argv=sys.argv[2:];runpy.run_module(sys.argv[0],run_name="__main__")')
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-c', launcher, str(output/'source'), MODULE,
                                '--worker', str(file)], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            record = load(Path(job['target']).with_suffix('.json'))
            if (record['job_sha256'] != digest(file) or record['status'] != 'complete'
                    or record['arrays_sha256'] != digest(Path(job['target']).with_suffix('.npz'))
                    or Path(record['source_file']) != output/'source'/Path(__file__).relative_to(ROOT)
                    or any(record[k] != job[k] for k in ('stage', 'subject', 'arm', 'point', 'seed', 'particles'))
                    or record['trials'] != job['training_trials'] + job.get('test_trials', 0)):
                raise ValueError('Worker output is incomplete or changed')
        print(f'{jobs[0]["stage"]}: {len(jobs)} jobs; {min(workers,len(jobs))} workers', flush=True)
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for i, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
                future.result()
                if i % 8 == 0 or i == len(jobs):
                    print(f'{i}/{len(jobs)} complete', flush=True)

    def training_stage(stage, cells):
        jobs = [make_job(s, arm, p, stage, seed) for (s, arm), ps in cells.items()
                for p in ps for seed in seeds[s][stage]]
        execute(jobs)
        rows = []
        for (s, arm), ps in cells.items():
            for p in ps:
                matched = [j for j in jobs if j['subject'] == s and j['arm'] == arm and j['point'] == p]
                probabilities = []
                for j in matched:
                    with np.load(Path(j['target']).with_suffix('.npz')) as z:
                        probabilities.append(z['observed_probability'])
                bank = np.stack(probabilities)
                rows.append(dict(subject=s, arm=arm, id=p['id'], **p['parameters'], nll=nll(bank),
                    mc_se=jackknife_se(jackknife_scores(bank)) if len(bank) > 1 else None,
                    filters=len(bank), particles=matched[0]['particles']))
        pd.DataFrame(rows).to_csv(output/(stage+'.csv'), index=False)
        return rows

    cells = {(s, arm): selected_points for s in subjects for arm in ARMS}
    screen = training_stage('screen', cells)
    finalists = {}
    for cell, ps in cells.items():
        rows = [r for r in screen if (r['subject'], r['arm']) == cell]
        ids = finalist_ids(rows, anchor['id'], 1 if smoke else settings['finalists'])
        finalists[cell] = [p for p in ps if p['id'] in ids]
    write(output/'finalists.json', dict(selection='screen training NLL',
        cells=[dict(subject=s, arm=arm, points=ps) for (s, arm), ps in finalists.items()]))
    confirmed = training_stage('confirmation', finalists)
    nominees = []
    for (s, arm), ps in finalists.items():
        best = min((r for r in confirmed if (r['subject'], r['arm']) == (s, arm)), key=lambda r: (r['nll'], r['id']))
        nominees.append(dict(subject=s, arm=arm, point=next(p for p in ps if p['id'] == best['id']), confirmation=best))
    write(output/'nominees.json', dict(status='frozen_before_later_choices', smoke=smoke, nominees=nominees,
        protocol_sha256=digest(output/'protocol.json'), confirmation_sha256=digest(output/'confirmation.csv')))
    evaluation = [make_job(n['subject'], n['arm'], n['point'], stage, seed) for n in nominees
                  for stage in ('evaluation', 'autonomous') for seed in seeds[n['subject']][stage]]
    execute(evaluation)
    rows = []
    for n in nominees:
        s, arm = n['subject'], n['arm']
        filters, trajectories = [], []
        for j in evaluation:
            if (j['subject'], j['arm']) != (s, arm):
                continue
            with np.load(Path(j['target']).with_suffix('.npz')) as z:
                if j['stage'] == 'evaluation':
                    filters.append(z['observed_probability'])
                else:
                    np.testing.assert_array_equal(z['generated_feedback'], z['generated_choices'] == z['categories'])
                    trajectories.append(z['generated_feedback'])
        probabilities, correct = np.stack(filters), np.stack(trajectories)
        cut = 8 if smoke else metadata[s]['training_trials']
        for window, a, b in [('training', 0, cut), ('later', cut, correct.shape[1]),
                             ('last128', max(0, correct.shape[1]-128), correct.shape[1])]:
            rows.append(dict(subject=s, arm=arm, id=n['point']['id'], window=window,
                nll=nll(probabilities[:, a:b]),
                nll_mc_se=jackknife_se(jackknife_scores(probabilities[:, a:b])) if len(filters) > 1 else None,
                autonomous_accuracy=float(correct[:, a:b].mean()),
                autonomous_mc_se=float(correct[:, a:b].mean(1).std(ddof=1)/np.sqrt(len(correct))) if len(correct)>1 else None))
    pd.DataFrame(rows).to_csv(output/'evaluation.csv', index=False)
    verify(inputs)
    artifacts = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                 if p.is_file() and 'source' not in p.relative_to(output).parts and p.suffix != '.log'}
    write(output/'completion.json', dict(status='complete', smoke=smoke, artifacts_sha256=artifacts,
        protocol_sha256=digest(output/'protocol.json'), nomination_sha256=digest(output/'nominees.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Supply --worker or both --config and --output')
