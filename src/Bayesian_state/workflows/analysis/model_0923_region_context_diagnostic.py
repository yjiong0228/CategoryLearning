"""Isolate local rule evidence at previously frozen behavior parameters.

Use --config YAML --output NEW_DIRECTORY [--smoke]. GLOBAL reproduces the
completed independent-row learner; LOCAL changes only its rule evidence readout.
Both use the original ROW beta/gamma/rate and the same first declared path seeds.
All 152 geometries stay available. Choice/feedback histories are observed for
evaluation and generated independently for autonomous paths. Oral input is absent.
This fixed-parameter causal diagnostic precedes refitting or finite-H admission.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...metrics.task import category_learning_feedback
from ...model.region_context_evidence import ContextCatalogueRegionLearner
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION
from .model_0923_h_validation import write
from .model_0923_region_association_diagnostic import ROOT, verify

MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_region_context_diagnostic'


def worker(file: Path) -> None:
    job = load(file); verify(job['bindings'])
    with np.load(job['geometry']) as z:
        q, categories = z['geometry'][:job['length']], z['categories'][:job['length']]
    if job['stage'] == 'evaluation':
        # The original observed-data fit and autonomous cache used different
        # stimulus batch lengths. Preserve the fit's exact geometry arithmetic.
        original_job = load(Path(job['reference_job']))
        with np.load(original_job['geometry']) as z:
            q = z['geometry'][original_job['beta_index'], :original_job['training_length']]
        if job['length'] > len(q):
            with np.load(original_job['later']) as z: x = z['stimulus'][:original_job['test_length']]
            partition = ContinuousPartition(4, 4, structural_extension=RELATIONAL_EXTENSION)
            later = np.stack([partition.get_category_probabilities(h, (x,), job['beta'],
                              distance_mode='boundary').T for h in range(len(q[0]))], axis=1)
            q = np.concatenate([q, later])
        q = q[:job['length']]
    model = ContextCatalogueRegionLearner(len(q[0]), rule_retention=job['gamma'],
        association_learning_rate=job['rate'], mode=job['arm'])
    probability, choices, feedback, weights, associations = [], [], [], [], []
    if job['stage'] == 'evaluation':
        parts = []
        for path in job['data']:
            with np.load(path) as z: parts.append({k: z[k] for k in ('choices', 'feedback', 'categories')})
        data = {k: np.concatenate([p[k] for p in parts])[:len(q)] for k in parts[0]}
        np.testing.assert_array_equal(data['categories'], categories)
    else:
        rng = np.random.default_rng(job['seed'])
    for t, geometry in enumerate(q):
        p = model.predict(geometry)
        if job['stage'] == 'evaluation':
            y, f = int(data['choices'][t]), float(data['feedback'][t])
            weights.append(model.rule_weights(geometry)); associations.append(model.associations)
        elif job['stage'] == 'autonomous':
            y = int(rng.choice(4, p=p)+1)
            f = category_learning_feedback(y, int(categories[t]), condition=2)
        else:
            raise ValueError('Unknown context diagnostic stage')
        probability.append(p); choices.append(y); feedback.append(f)
        model.update(geometry, y, f)
    arrays = dict(probabilities=np.asarray(probability), choices=np.asarray(choices),
                  feedback=np.asarray(feedback), categories=categories)
    if job['stage'] == 'evaluation':
        arrays.update(rule_weights=np.asarray(weights), associations=np.asarray(associations),
            geometry=q, observed_probability=arrays['probabilities'][np.arange(len(q)), arrays['choices']-1])
    original_elements = 0
    if job['arm'] == 'GLOBAL':
        with np.load(job['reference']) as reference:
            fields = ('probabilities', 'rule_weights', 'associations') if job['stage'] == 'evaluation' else ('probabilities', 'choices', 'feedback')
            for key in fields:
                original = reference[key]
                if job['stage'] == 'autonomous': original = original[job['reference_index']]
                np.testing.assert_array_equal(arrays[key], original[:len(q)], err_msg=key)
                original_elements += arrays[key].size
    p = arrays['probabilities']
    if p.shape != (len(q), 4) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid choice prediction')
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(arrays['feedback'], (arrays['choices'] == categories).astype(float))
    verify(job['bindings']); target = Path(job['target'])
    with target.with_suffix('.npz').open('xb') as handle: np.savez_compressed(handle, **arrays)
    write(target.with_suffix('.json'), dict(status='complete', subject=job['subject'], arm=job['arm'],
        stage=job['stage'], trials=len(q), seed=job.get('seed'), original_elements_exact=original_elements,
        job_sha256=digest(file), arrays_sha256=digest(target.with_suffix('.npz'))))


def run(config: Path, output: Path, smoke: bool = False) -> None:
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'reference_fit', 'reference_autonomous', 'paths', 'workers'}
            or type(settings['paths']) is not int or not 4 <= settings['paths'] <= 128
            or type(settings['workers']) is not int or not 1 <= settings['workers'] <= 64):
        raise ValueError('Declare completed references, 4..128 paths and at most64 workers')
    if output.exists(): raise FileExistsError('Use a new context diagnostic directory')
    output = output.resolve(); fit = ROOT/settings['reference_fit']; auto = ROOT/settings['reference_autonomous']
    fd, fp, nominees = [load(fit/name) for name in ('completion.json', 'protocol.json', 'nominees.json')]
    ad, ap = [load(auto/name) for name in ('completion.json', 'protocol.json')]
    if (fd['status'] != 'complete' or ad['status'] != 'complete' or fd['smoke'] or ad['smoke']
            or fd['nomination_sha256'] != digest(fit/'nominees.json')
            or ap['parent_nomination_sha256'] != fd['nomination_sha256']
            or nominees['status'] != 'frozen_before_later_choices'):
        raise ValueError('Require completed full references with frozen parameters')
    bindings = dict(fp['bindings'])
    for p, h in ap['bindings'].items(): bindings[p] = h
    files = [Path(__file__), ROOT/'src/Bayesian_state/model/region_context_evidence.py',
        ROOT/'src/Bayesian_state/metrics/autonomous_fit.py', ROOT/'src/Bayesian_state/metrics/task.py']
    for p in [config, *files, fit/'completion.json', fit/'nominees.json', fit/'protocol.json',
              auto/'completion.json', auto/'protocol.json']:
        bindings[str(p.resolve())] = digest(p)
    points = [n for n in nominees['nominees'] if n['arm'] == 'ROW']
    if smoke: points = points[:1]
    geometries, old_paths = {}, {}
    for p in sorted((auto/'jobs').glob('*.json')):
        job = load(p); target = Path(job['target']).with_suffix('.npz'); record = load(target.with_suffix('.json'))
        if record['job_sha256'] != digest(p) or record['arrays_sha256'] != digest(target):
            raise ValueError('Autonomous reference changed')
        if job['stage'] == 'geometry': geometries[job['subject'], job['beta']] = target
        elif job['arm'] == 'ROW':
            for i, seed in enumerate(job['seeds']): old_paths[job['subject'], seed] = target, i
    jobs, metadata, seed_banks = [], {}, {}
    for point in points:
        s = point['subject']; folder = ROOT/fp['settings']['data']/f'subject_{s}'
        meta = load(folder/'prepared.json'); metadata[s] = meta
        if meta['condition'] != 2: raise ValueError('Context diagnostic supports condition2 only')
        for name, h in meta['array_sha256'].items(): bindings[str(folder/name)] = h
        bindings[str(folder/'prepared.json')] = digest(folder/'prepared.json')
        geometry = geometries[s, point['beta']]; bindings[str(geometry)] = digest(geometry)
        original = fit/'evaluation'/f'S{s}_ROW.npz'; record = load(original.with_suffix('.json'))
        bindings[str(original)] = record['array_sha256']; bindings[str(original.with_suffix('.json'))] = digest(original.with_suffix('.json'))
        reference_job = fit/'jobs'/f'S{s}_ROW_evaluation.json'
        bindings[str(reference_job)] = record['job_sha256']
        original_job = load(reference_job)
        bindings[original_job['geometry']] = digest(Path(original_job['geometry']))
        seeds = ap['seed_banks'][str(s)][:4 if smoke else settings['paths']]; seed_banks[s] = seeds
        if len(seeds) != (4 if smoke else settings['paths']): raise ValueError('Incomplete original seed bank')
        length = min(16, meta['trials']) if smoke else meta['trials']
        for arm in ('GLOBAL', 'LOCAL'):
            common = dict(subject=s, arm=arm, beta=point['beta'], gamma=point['gamma'], rate=point['rate'],
                          geometry=str(geometry), length=length)
            name = f'evaluation_S{s}_{arm}'
            jobs.append(dict(common, name=name, stage='evaluation', data=[str(folder/n) for n in ('training.npz', 'sealed_test.npz')],
                             reference=str(original), reference_job=str(reference_job), target=str(output/'chunks'/name)))
            for seed in seeds:
                ref, index = old_paths[s, seed]; bindings[str(ref)] = digest(ref)
                name = f'autonomous_S{s}_{arm}_{seed}'
                jobs.append(dict(common, name=name, stage='autonomous', seed=seed, reference=str(ref),
                                 reference_index=index, target=str(output/'chunks'/name)))
    verify(bindings); output.mkdir(parents=True); (output/'chunks').mkdir(); (output/'jobs').mkdir()
    for p in files:
        target = output/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    shutil.copyfile(config, output/'config.yaml')
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, points=points, seeds=seed_banks,
        bindings=bindings, scope='Fixed original ROW parameters,152 always available rules; only evidence readout changes',
        oral_used=False, parameter_selection='Original training-only ROW nominees; no new fitting',
        scoring='Equal-weight paths; 32-trial accuracy and category-conditioned response frequencies, equal family weights',
        python=sys.version, versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    workers = min(1 if smoke else settings['workers'], len(jobs), len(os.sched_getaffinity(0)))

    def launch(job):
        job = dict(job, bindings=bindings); file = output/'jobs'/(job['name']+'.json'); write(file, job)
        with tempfile.TemporaryDirectory(prefix='region_context_') as cache, file.with_suffix('.log').open('x') as log:
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=cache, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-m', MODULE, '--worker', str(file)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return load(Path(job['target']).with_suffix('.json'))

    print(f'{len(jobs)} context jobs; {workers} single-thread workers', flush=True); records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            records.append(f.result())
            if i % 16 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)
    scores = []
    for point in points:
        s = point['subject']; meta = metadata[s]; length = min(16, meta['trials']) if smoke else meta['trials']
        cut = min(meta['training_trials'], length)
        for arm in ('GLOBAL', 'LOCAL'):
            with np.load(output/'chunks'/f'evaluation_S{s}_{arm}.npz') as z:
                p, human, labels = z['observed_probability'], z['choices'], z['categories']
            choices, feedback = [], []
            for seed in seed_banks[s]:
                with np.load(output/'chunks'/f'autonomous_S{s}_{arm}_{seed}.npz') as z:
                    choices.append(z['choices']); feedback.append(z['feedback'])
            choices, feedback = np.stack(choices), np.stack(feedback)
            for window, a, b in [('training', 0, cut), ('later', cut, length), ('last128', max(0, length-128), length)]:
                if a == b: continue
                features = behavior_features(choices[:, a:b], labels[a:b], n_categories=4)
                target = behavior_features(human[None, a:b], labels[a:b], n_categories=4)[0]
                energy = energy_u(features, target)
                scores.append(dict(subject=s, arm=arm, window=window, trials=b-a,
                    nll=float(-np.log(p[a:b]).mean()), autonomous_accuracy=float(feedback[:, a:b].mean()),
                    accuracy_mcse=float(feedback[:, a:b].mean(1).std(ddof=1)/len(feedback)**.5),
                    energy_u=energy['value'], energy_u_mcse=energy['mc_se']))
    pd.DataFrame(scores).to_csv(output/'scores.csv', index=False)
    verify(bindings)
    write(output/'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        original_elements_exact=sum(r['original_elements_exact'] for r in records),
        protocol_sha256=digest(output/'protocol.json'), bindings=bindings,
        artifacts={str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                   if p.is_file() and 'source' not in p.relative_to(output).parts and p.suffix != '.log'}))
    print(pd.DataFrame(scores).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Supply --worker or both --config and --output')
