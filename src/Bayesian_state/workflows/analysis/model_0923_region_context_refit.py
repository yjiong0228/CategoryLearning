"""Refit local evidence on the original independent-row behavior grid.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Each full LOCAL fit uses
the original GLOBAL beta/gamma/rate grid and training boundary. Completed GLOBAL
scores and paths are checked and reused. Nominees are saved before later choices
are opened. Later feedback continues learning; autonomous paths generate their
own choices and feedback. Oral reports are absent from this workflow.

Smoke uses the first subject, four grid points, eight training and eight later
trials, four paths and one worker. It does not compare with full GLOBAL results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
from itertools import product
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
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...metrics.task import category_learning_feedback
from ...model.region_context_evidence import ContextCatalogueRegionLearner
from .model_0923_h_validation import write
from .model_0923_region_association_diagnostic import ROOT, verify

MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_region_context_refit'


def replay(q: np.ndarray, choices: np.ndarray, feedback: np.ndarray, *,
           gamma: float, rate: float, trace: bool = False) -> dict:
    """Use completed feedback only; retain pre-feedback states for oral evaluation."""
    q, y, f = np.asarray(q), np.asarray(choices), np.asarray(feedback)
    if (q.ndim != 3 or q.shape[2] != 4 or not len(q) or y.shape != (len(q),)
            or f.shape != y.shape or not np.isin(y, [1, 2, 3, 4]).all()
            or not np.isin(f, [0., 1.]).all()):
        raise ValueError('Require aligned four-choice binary-feedback trials')
    model = ContextCatalogueRegionLearner(q.shape[1], rule_retention=gamma,
        association_learning_rate=rate, mode='LOCAL')
    p, weights, associations = [], [], []
    for geometry, choice, outcome in zip(q, y, f):
        p.append(model.predict(geometry))
        if trace:
            weights.append(model.rule_weights(geometry)); associations.append(model.associations)
        model.update(geometry, int(choice), float(outcome))
    p = np.asarray(p)
    arrays = dict(probabilities=p, observed_probability=p[np.arange(len(q)), y.astype(int)-1])
    if trace: arrays.update(rule_weights=np.asarray(weights), associations=np.asarray(associations))
    return arrays


def worker(file: Path) -> None:
    job = load(file); verify(job['bindings'])
    stage = job['stage']
    if stage == 'autonomous':
        with np.load(job['evaluation']) as z: q, categories = z['autonomous_geometry'], z['categories']
        model = ContextCatalogueRegionLearner(q.shape[1], rule_retention=job['gamma'],
            association_learning_rate=job['rate'], mode='LOCAL')
        rng = np.random.default_rng(job['seed']); probabilities, choices, feedback = [], [], []
        for geometry, category in zip(q, categories):
            p = model.predict(geometry); y = int(rng.choice(4, p=p)+1)
            f = category_learning_feedback(y, int(category), condition=2)
            probabilities.append(p); choices.append(y); feedback.append(f)
            model.update(geometry, y, f)
        arrays = dict(probabilities=np.asarray(probabilities), choices=np.asarray(choices),
                      feedback=np.asarray(feedback), categories=categories)
    else:
        with np.load(job['geometry']) as z: q = z['geometry'][job['beta_index'], :job['training_length']]
        with np.load(job['training']) as z:
            data = {k: z[k][:len(q)] for k in ('stimulus', 'choices', 'feedback', 'categories')}
        cut = len(q)
        if stage == 'evaluation':
            frozen = load(Path(job['nominees']))
            expected = {k: job[k] for k in ('subject', 'arm', 'beta', 'gamma', 'rate')}
            if (frozen['status'] != 'frozen_before_later_choices'
                    or not any(all(n[k] == v for k, v in expected.items()) for n in frozen['nominees'])):
                raise ValueError('Evaluation parameters must match the frozen training nominee')
            with np.load(job['later']) as z:
                later = {k: z[k][:job['test_length']] for k in data}
            partition = ContinuousPartition(4, 4, structural_extension=RELATIONAL_EXTENSION)
            def geometry(x):
                return np.stack([partition.get_category_probabilities(h, (x,), job['beta'],
                    distance_mode='boundary').T for h in range(152)], axis=1)
            q = np.concatenate([q, geometry(later['stimulus'])])
            data = {k: np.concatenate([data[k], later[k]]) for k in data}
            auto_q = geometry(data['stimulus'])
            np.testing.assert_allclose(q, auto_q, atol=1e-12, rtol=0)
        arrays = replay(q, data['choices'], data['feedback'], gamma=job['gamma'], rate=job['rate'],
                        trace=stage == 'evaluation')
        if stage == 'evaluation':
            with np.load(job['training_prediction']) as z:
                np.testing.assert_array_equal(arrays['probabilities'][:cut], z['probabilities'])
            arrays.update(geometry=q, autonomous_geometry=auto_q,
                          **{k: data[k] for k in ('choices', 'feedback', 'categories')})
    p = arrays['probabilities']
    if not np.isfinite(p).all() or np.any(p < 0): raise ValueError('Invalid choice probabilities')
    np.testing.assert_allclose(p.sum(-1), 1., atol=1e-12, rtol=0)
    if 'feedback' in arrays:
        np.testing.assert_array_equal(arrays['feedback'], (arrays['choices'] == arrays['categories']).astype(float))
    record = {k: job[k] for k in ('subject', 'arm', 'beta', 'gamma', 'rate', 'stage')}
    record.update(status='complete', trials=len(p), seed=job.get('seed'))
    if stage != 'autonomous':
        observed = arrays['observed_probability']
        if np.any(observed <= 0): raise ValueError('Nonpositive observed probability')
        record['mean_nll'] = float(-np.log(observed).mean())
        if stage == 'evaluation':
            record.update(training_nll=float(-np.log(observed[:cut]).mean()),
                          test_nll=float(-np.log(observed[cut:]).mean()), exact_training_prefix=True)
    verify(job['bindings']); target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle: np.savez_compressed(handle, **arrays)
    record.update(arrays_sha256=digest(target.with_suffix('.npz')), job_sha256=digest(file))
    write(target.with_suffix('.json'), record)


def execute(jobs: list[dict], output: Path, workers: int) -> None:
    workers = min(workers, len(jobs), len(os.sched_getaffinity(0)))
    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        with tempfile.TemporaryDirectory(prefix='context_refit_') as cache, file.with_suffix('.log').open('x') as log:
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-m', MODULE, '--worker', str(file)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'{jobs[0]["stage"]}: {len(jobs)} jobs, {workers} single-thread workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if i % 16 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'reference_fit', 'reference_fixed', 'paths', 'workers'}
            or type(settings['workers']) is not int or not 1 <= settings['workers'] <= 64
            or type(settings['paths']) is not int or not 4 <= settings['paths'] <= 32):
        raise ValueError('Declare both completed references,4–32 paths and1–64 workers')
    output = output.resolve()
    if output.exists(): raise FileExistsError('Use a new context refit output directory')
    fit, fixed = ROOT/settings['reference_fit'], ROOT/settings['reference_fixed']
    fp, fd, fn = [load(fit/n) for n in ('protocol.json', 'completion.json', 'nominees.json')]
    xp, xd = [load(fixed/n) for n in ('protocol.json', 'completion.json')]
    if (fd['status'] != 'complete' or xd['status'] != 'complete' or fd['smoke'] or xd['smoke']
            or fd['nomination_sha256'] != digest(fit/'nominees.json')
            or fd['protocol_sha256'] != digest(fit/'protocol.json')
            or xd['protocol_sha256'] != digest(fixed/'protocol.json')
            or (ROOT/xp['settings']['reference_fit']).resolve() != fit.resolve()):
        raise ValueError('Require completed full, mutually bound references')
    bindings = dict(fp['bindings'])
    files = [Path(__file__), ROOT/'src/Bayesian_state/model/region_context_evidence.py',
             ROOT/'src/Bayesian_state/metrics/autonomous_fit.py', ROOT/'src/Bayesian_state/metrics/task.py']
    for p in [config, *files, *[fit/n for n in ('protocol.json', 'completion.json', 'nominees.json')],
              *[fixed/n for n in ('protocol.json', 'completion.json', 'scores.csv')]]:
        bindings[str(p.resolve())] = digest(p)
    subjects = fp['subjects'][:1] if smoke else fp['subjects']
    expected_grid = set(product(fp['actual_beta'], fp['actual_gamma'], fp['actual_rates']))
    original, reused = [], []
    for s in subjects:
        selected = []
        for file in sorted((fit/'jobs').glob(f'S{s}_ROW_B*.json')):
            j = load(file); target = Path(j['target']); record = load(target.with_suffix('.json'))
            if record['job_sha256'] != digest(file) or record['array_sha256'] != digest(target.with_suffix('.npz')):
                raise ValueError('Original training candidate changed')
            if j['bindings'] != fp['bindings']: raise ValueError('Original candidate provenance differs')
            with np.load(target.with_suffix('.npz')) as z: score = float(-np.log(z['observed_probability']).mean())
            if score != record['mean_nll']: raise ValueError('Original training score differs')
            for p in (file, target.with_suffix('.json'), target.with_suffix('.npz')): bindings[str(p)] = digest(p)
            selected.append(j); reused.append(dict(subject=s, arm='GLOBAL', beta=j['beta'], gamma=j['gamma'],
                rate=j['rate'], mean_nll=score, trials=record['trials']))
        if (len(selected) != len(expected_grid)
                or {(j['beta'], j['gamma'], j['rate']) for j in selected} != expected_grid):
            raise ValueError('Original grid is incomplete or contains duplicate candidates')
        original.extend(selected)
    verify(bindings); output.mkdir(parents=True); (output/'jobs').mkdir(); (output/'chunks').mkdir()
    for path, expected in bindings.items():
        p = Path(path)
        if p.suffix == '.py' and p.is_relative_to(ROOT/'src'):
            target = output/'source'/p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    shutil.copyfile(config, output/'config.yaml')
    jobs = []
    for j in original:
        if smoke and (j['beta'] != fp['actual_beta'][0] or j['gamma'] not in fp['actual_gamma'][:2]
                      or j['rate'] not in fp['actual_rates'][:2]): continue
        name = j['name'].replace('_ROW_', '_LOCAL_')
        jobs.append(dict(j, name=name, arm='LOCAL', target=str(output/'training'/name), bindings=bindings,
                         training_length=min(8, j['training_length']) if smoke else j['training_length']))
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, bindings=bindings, subjects=subjects,
        grid=dict(beta=fp['actual_beta'], gamma=fp['actual_gamma'], rate=fp['actual_rates']),
        local_training_jobs=len(jobs), global_full_training_points_reused=0 if smoke else len(reused),
        selection='Training choice NLL; ties beta,gamma,rate; parameters frozen before later choices',
        scope='152 always available rules; independent association learning; only evidence readout differs',
        oral_used=False, seeds={str(s): xp['seeds'][str(s)][:4 if smoke else settings['paths']] for s in subjects},
        python=sys.version, versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    workers = 1 if smoke else settings['workers']; execute(jobs, output, workers)
    local = [load(Path(j['target']).with_suffix('.json')) for j in jobs]
    cols = ['subject', 'arm', 'beta', 'gamma', 'rate', 'trials', 'mean_nll']
    scores = pd.DataFrame(local + ([] if smoke else reused))[cols]
    scores.to_csv(output/'training_scores.csv', index=False); nominees = []
    for (s, arm), group in scores.groupby(['subject', 'arm'], sort=True):
        best = group.sort_values(['mean_nll', 'beta', 'gamma', 'rate']).iloc[0].to_dict()
        best.update(actual_candidates=len(group), ties=int(group.mean_nll.eq(best['mean_nll']).sum()))
        if arm == 'GLOBAL':
            old = next(n for n in fn['nominees'] if n['subject'] == s and n['arm'] == 'ROW')
            if any(best[k] != old[k] for k in ('beta', 'gamma', 'rate', 'mean_nll', 'actual_candidates')):
                raise ValueError('Reconstructed GLOBAL nominee differs')
        nominees.append(best)
    write(output/'nominees.json', dict(status='frozen_before_later_choices', nominees=nominees,
        protocol_sha256=digest(output/'protocol.json'), training_scores_sha256=digest(output/'training_scores.csv')))
    evaluations = []
    for n in nominees:
        if n['arm'] != 'LOCAL': continue
        j = next(j for j in jobs if all(j[k] == n[k] for k in ('subject', 'beta', 'gamma', 'rate')))
        folder = Path(j['training']).parent; meta = load(folder/'prepared.json'); later = folder/'sealed_test.npz'
        if meta['condition'] != 2: raise ValueError('Only condition2 is supported')
        name = f'evaluation_S{n["subject"]}_LOCAL'
        evaluations.append(dict(j, name=name, stage='evaluation', target=str(output/'chunks'/name),
            nominees=str(output/'nominees.json'), later=str(later),
            test_length=min(8, meta['test_trials']) if smoke else meta['test_trials'],
            training_prediction=str(Path(j['target']).with_suffix('.npz')),
            bindings={**bindings, str(later): meta['array_sha256']['sealed_test.npz'],
                      str(output/'nominees.json'): digest(output/'nominees.json'),
                      str(Path(j['target']).with_suffix('.npz')): digest(Path(j['target']).with_suffix('.npz'))}))
    execute(evaluations, output, workers); paths = []; seed_banks = load(output/'protocol.json')['seeds']
    for j in evaluations:
        p = Path(j['target']).with_suffix('.npz')
        for seed in seed_banks[str(j['subject'])]:
            name = f'autonomous_S{j["subject"]}_LOCAL_{seed}'
            paths.append(dict(j, name=name, stage='autonomous', seed=seed, evaluation=str(p),
                target=str(output/'chunks'/name), bindings={**j['bindings'], str(p): digest(p)}))
    execute(paths, output, workers); result = []
    for j in evaluations:
        s = j['subject']; cut = j['training_length']
        with np.load(Path(j['target']).with_suffix('.npz')) as z:
            observed, human, categories = z['observed_probability'], z['choices'], z['categories']
        choices, feedback = [], []
        for seed in seed_banks[str(s)]:
            with np.load(output/'chunks'/f'autonomous_S{s}_LOCAL_{seed}.npz') as z:
                choices.append(z['choices']); feedback.append(z['feedback'])
        choices, feedback = np.stack(choices), np.stack(feedback)
        for window, a, b in [('training', 0, cut), ('later', cut, len(human)), ('last128', max(0, len(human)-128), len(human))]:
            features = behavior_features(choices[:, a:b], categories[a:b], n_categories=4)
            target = behavior_features(human[None, a:b], categories[a:b], n_categories=4)[0]
            energy = energy_u(features, target)
            result.append(dict(subject=s, arm='LOCAL', window=window, trials=b-a,
                nll=float(-np.log(observed[a:b]).mean()), autonomous_accuracy=float(feedback[:, a:b].mean()),
                accuracy_mcse=float(feedback[:, a:b].mean(1).std(ddof=1)/len(feedback)**.5),
                energy_u=energy['value'], energy_u_mcse=energy['mc_se']))
    result = pd.DataFrame(result)
    if not smoke and settings['paths'] == xp['settings']['paths']:
        baseline = pd.read_csv(fixed/'scores.csv'); result = pd.concat([baseline[baseline.arm == 'GLOBAL'], result])
    result.to_csv(output/'scores.csv', index=False); verify(bindings)
    write(output/'completion.json', dict(status='complete', smoke=smoke, training_jobs=len(jobs),
        reused_global_training_points=0 if smoke else len(reused), evaluations=len(evaluations), paths=len(paths),
        protocol_sha256=digest(output/'protocol.json'), nomination_sha256=digest(output/'nominees.json'),
        artifacts={str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                   if p.is_file() and 'source' not in p.relative_to(output).parts and p.suffix != '.log'}))
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Supply --worker or both --config and --output')
